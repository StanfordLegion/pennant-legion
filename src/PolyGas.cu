
#include "PolyGas.hh"
#include "Mesh.hh"
#include "MyLegion.hh"
#include "CudaHelp.hh"

using namespace Legion;

namespace {  // unnamed
__host__
static void __attribute__ ((constructor)) registerTasks() {
  {
      TaskVariantRegistrar registrar(TID_CALCSTATEHALF, "GPU calcstatehalf");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<PolyGas::calcStateHalfGPUTask>(registrar, "calcstatehalf");
  }
  {
      TaskVariantRegistrar registrar(TID_CALCFORCEPGAS, "GPU calcforcepgas");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<PolyGas::calcForceGPUTask>(registrar, "calcforcepgas");
    }
}
}; // namespace

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_state_half(const AccessorRO<double> acc_zr,
               const AccessorRO<double> acc_zvolp,
               const AccessorRO<double> acc_zvol0,
               const AccessorRO<double> acc_ze,
               const AccessorRO<double> acc_zwrate,
               const AccessorRO<double> acc_zm,
               const AccessorWD<double> acc_zp,
               const AccessorWD<double> acc_zss,
               const double dth, const double gm1, const double ssmin2,
               const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t z = origin[0] + offset;
  // compute EOS at beginning of time step
  const double r = acc_zr[z];
  const double ze = acc_ze[z];
  const double e = (ze > 0.) ? ze : 0.;
  const double p = gm1 * r * e;
  const double pre = gm1 * e;
  const double per = gm1 * r;
  const double mt = pre + per * p / (r * r);
  const double csqd = (ssmin2 > mt) ? ssmin2 : mt;
  const double ss = sqrt(csqd);

  // now advance pressure to the half-step
  const double minv = 1. / acc_zm[z];
  const double volp = acc_zvolp[z];
  const double vol0 = acc_zvol0[z];
  const double wrate = acc_zwrate[z];
  const double dv = (volp - vol0) * minv;
  const double bulk = r * csqd;
  const double denom = 1. + 0.5 * per * dv;
  const double src = wrate * dth * minv;
  acc_zp[z] = p + (per * src - r * bulk * dv) / denom;
  acc_zss[z] = ss;
}

__host__
void PolyGas::calcStateHalfGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double gamma = args[0];
    const double ssmin = args[1];
    const double dt    = task->futures[0].get_result<double>();

    const AccessorRO<double> acc_zr(regions[0], FID_ZR);
    const AccessorRO<double> acc_zvolp(regions[0], FID_ZVOLP);
    const AccessorRO<double> acc_zvol0(regions[0], FID_ZVOL0);
    const AccessorRO<double> acc_ze(regions[0], FID_ZE);
    const AccessorRO<double> acc_zwrate(regions[0], FID_ZWRATE);
    const AccessorRO<double> acc_zm(regions[0], FID_ZM);
    const AccessorWD<double> acc_zp(regions[1], FID_ZP);
    const AccessorWD<double> acc_zss(regions[1], FID_ZSS);

    const double dth = 0.5 * dt;
    const double gm1 = gamma - 1.;
    const double ssmin2 = max(ssmin * ssmin, 1.e-99);
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    const size_t volume = rectz.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_state_half<<<blocks,THREADS_PER_BLOCK>>>(acc_zr, acc_zvolp, acc_zvol0,
        acc_ze, acc_zwrate, acc_zm, acc_zp, acc_zss, dth, gm1, ssmin2,
        rectz.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_force_pgas(const AccessorRO<Pointer> acc_mapsz,
                    const AccessorRO<double2> acc_ssurf,
                    const AccessorRO<double> acc_zp,
                    const AccessorWD<double2> acc_sf,
                    const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer z = acc_mapsz[s];
  const double p = acc_zp[z];
  const double2 surf = acc_ssurf[s];
  const double2 sfx = -p * surf;
  acc_sf[s] = sfx;
}

__host__
void PolyGas::calcForceGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double2> acc_ssurf(regions[0], FID_SSURFP);
    const AccessorRO<double> acc_zp(regions[1], FID_ZP);
    const AccessorWD<double2> acc_sf(regions[2], FID_SFP);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_force_pgas<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsz, acc_ssurf,
        acc_zp, acc_sf, rects.lo, volume);
}

