
#include "TTS.hh"
#include "Mesh.hh"
#include "Hydro.hh"
#include "MyLegion.hh"
#include "CudaHelp.hh"

using namespace Legion;

namespace {  // unnamed
__host__
static void __attribute__ ((constructor)) registerTasks() {
  {
    TaskVariantRegistrar registrar(TID_CALCFORCETTS, "GPU calcforcetts");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<TTS::calcForceGPUTask>(registrar, "calcforcetts");
  }
}
}; // namespace

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_force_tts(const AccessorRO<Pointer> acc_mapsz,
                   const AccessorRO<double> acc_sarea,
                   const AccessorRO<double> acc_smf,
                   const AccessorRO<double2> acc_ssurf,
                   const AccessorRO<double> acc_zarea,
                   const AccessorRO<double> acc_zr,
                   const AccessorRO<double> acc_zss,
                   const AccessorWD<double2> acc_sf,
                   const double alfa, const double ssmin,
                   const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer z = acc_mapsz[s];
  const double sarea = acc_sarea[s];
  const double zarea = acc_zarea[z];
  const double vfacinv = zarea / sarea;
  const double r = acc_zr[z];
  const double mf = acc_smf[s];
  const double srho = r * mf * vfacinv;
  const double ss = acc_zss[z];
  double sstmp = (ss > ssmin) ? ss : ssmin;
  sstmp = alfa * sstmp * sstmp;
  const double sdp = sstmp * (srho - r);
  const double2 surf = acc_ssurf[s];
  const double2 sqq = -sdp * surf;
  acc_sf[s] = sqq;
}

__host__
void TTS::calcForceGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double alfa  = args[0];
    const double ssmin = args[1];

    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double> acc_sarea(regions[0], FID_SAREAP);
    const AccessorRO<double> acc_smf(regions[0], FID_SMF);
    const AccessorRO<double2> acc_ssurf(regions[0], FID_SSURFP);
    const AccessorRO<double> acc_zarea(regions[1], FID_ZAREAP);
    const AccessorRO<double> acc_zr(regions[1], FID_ZRP);
    const AccessorRO<double> acc_zss(regions[1], FID_ZSS);
    const AccessorWD<double2> acc_sf(regions[2], FID_SFT);

    //  Side density:
    //    srho = sm/sv = zr (sm/zm) / (sv/zv)
    //  Side pressure:
    //    sp   = zp + alfa dpdr (srho-zr)
    //         = zp + sdp
    //  Side delta pressure:
    //    sdp  = alfa dpdr (srho-zr)
    //         = alfa c**2 (srho-zr)
    //
    //    Notes: smf stores (sm/zm)
    //           svfac stores (sv/zv)

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_force_tts<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsz, acc_sarea, 
        acc_smf, acc_ssurf, acc_zarea, acc_zr, acc_zss, acc_sf, alfa, ssmin,
        rects.lo, volume);
}

