
#include "QCS.hh"

#include "Vec2.hh"
#include "Mesh.hh"
#include "Hydro.hh"
#include "MyLegion.hh"
#include "CudaHelp.hh"

using namespace Legion;

namespace {  // unnamed
__host__
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_SETCORNERDIV, "GPU setcornerdiv");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setCornerDivGPUTask>(registrar, "setcornerdiv");
    }
    {
      TaskVariantRegistrar registrar(TID_SETQCNFORCE, "GPU setqcnforce");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setQCnForceGPUTask>(registrar, "setqcnforce");
    }
    {
      TaskVariantRegistrar registrar(TID_SETFORCEQCS, "GPU setforceqcs");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setForceGPUTask>(registrar, "setforceqcs");
    }
    {
      TaskVariantRegistrar registrar(TID_SETVELDIFF, "GPU setveldiff");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setVelDiffGPUTask>(registrar, "setveldiff");
    }
}
}; // namespace

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_zone_centered_velocity(const AccessorRO<Pointer> acc_mapsp1,
                           const AccessorRO<int> acc_mapsp1reg,
                           const AccessorRO<Pointer> acc_mapsz,
                           const AccessorRO<double2> acc_pu0,
                           const AccessorRO<double2> acc_pu1,
                           const AccessorRO<int> acc_znump,
                           const AccessorWD<double2> acc_zuc,
                           const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer p = acc_mapsp1[s];
  const int preg = acc_mapsp1reg[s];
  const Pointer z = acc_mapsz[s];
  const double2 pu = (preg == 0) ? acc_pu0[p] : acc_pu1[p];
  const int n = acc_znump[z];
  SumOp<double2>::apply<false/*exclusive*/>(acc_zuc[z], pu / n);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_corner_divergence(const AccessorRO<Pointer> acc_mapsz,
                      const AccessorRO<Pointer> acc_mapsp1,
                      const AccessorRO<Pointer> acc_mapsp2,
                      const AccessorRO<Pointer> acc_mapss3,
                      const AccessorRO<int> acc_mapsp1reg,
                      const AccessorRO<int> acc_mapsp2reg,
                      const AccessorRO<double2> acc_ex,
                      const AccessorRO<double> acc_elen,
                      const AccessorRO<int> acc_znump,
                      const AccessorRO<double2> acc_zx,
                      const AccessorRO<double2> acc_pu0,
                      const AccessorRO<double2> acc_pu1,
                      const AccessorRO<double2> acc_px0,
                      const AccessorRO<double2> acc_px1,
                      const AccessorWD<double2> acc_zuc,
                      const AccessorWD<double> acc_carea,
                      const AccessorWD<double> acc_ccos,
                      const AccessorWD<double> acc_cdiv,
                      const AccessorWD<double> acc_cevol,
                      const AccessorWD<double> acc_cdu,
                      const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t c = origin[0] + offset;
  const Pointer s2 = c;
  const Pointer s = acc_mapss3[s2];
  // Associated zone, point
  const Pointer z = acc_mapsz[s];
  const Pointer p = acc_mapsp2[s];
  const int preg = acc_mapsp2reg[s];
  // Neighboring points
  const Pointer p1 = acc_mapsp1[s];
  const int p1reg = acc_mapsp1reg[s];
  const Pointer p2 = acc_mapsp2[s2];
  const int p2reg = acc_mapsp2reg[s2];

  // Velocities and positions
  // 0 = point p
  const double2 up0 = (preg == 0) ? acc_pu0[p] : acc_pu1[p];
  const double2 xp0 = (preg == 0) ? acc_px0[p] : acc_px1[p];
  // 1 = edge e2
  const double2 up1 = 0.5 * (up0 + ((p2reg == 0) ? acc_pu0[p2] : acc_pu1[p2]));
  const double2 xp1 = acc_ex[s2];
  // 2 = zone center z
  const double2 up2 = acc_zuc[z];
  const double2 xp2 = acc_zx[z];
  // 3 = edge e1
  const double2 up3 = 0.5 * (((p1reg == 0) ? acc_pu0[p1] : acc_pu1[p1]) + up0);
  const double2 xp3 = acc_ex[s];

  // compute 2d cartesian volume of corner
  double cvolume = 0.5 * cross(xp2 - xp0, xp3 - xp1);
  acc_carea[c] = cvolume;

  // compute cosine angle
  const double2 v1 = xp3 - xp0;
  const double2 v2 = xp1 - xp0;
  const double de1 = acc_elen[s];
  const double de2 = acc_elen[s2];
  const double minelen = min(de1, de2);
  const double ccos = ((minelen < 1.e-12) ?
          0. :
          4. * dot(v1, v2) / (de1 * de2));
  acc_ccos[c] = ccos;

  // compute divergence of corner
  const double cdiv = (cross(up2 - up0, xp3 - xp1) -
          cross(up3 - up1, xp2 - xp0)) /
          (2.0 * cvolume);
  acc_cdiv[c] = cdiv;

  // compute evolution factor
  const double2 dxx1 = 0.5 * (xp1 + xp2 - xp0 - xp3);
  const double2 dxx2 = 0.5 * (xp2 + xp3 - xp0 - xp1);
  const double dx1 = length(dxx1);
  const double dx2 = length(dxx2);

  // average corner-centered velocity
  const double2 duav = 0.25 * (up0 + up1 + up2 + up3);

  const double test1 = abs(dot(dxx1, duav) * dx2);
  const double test2 = abs(dot(dxx2, duav) * dx1);
  const double num = (test1 > test2 ? dx1 : dx2);
  const double den = (test1 > test2 ? dx2 : dx1);
  const double r = num / den;
  double evol = sqrt(4.0 * cvolume * r);
  evol = min(evol, 2.0 * minelen);

  // compute delta velocity
  const double dv1 = length2(up1 + up2 - up0 - up3);
  const double dv2 = length2(up2 + up3 - up0 - up1);
  double du = sqrt((dv1 > dv2) ? dv1 : dv2);

  evol = (cdiv < 0.0 ? evol : 0.);
  du   = (cdiv < 0.0 ? du   : 0.);
  acc_cevol[c] = evol;
  acc_cdu[c] = du;
}

// Routine number [2]  in the full algorithm
//     [2.1] Find the corner divergence
//     [2.2] Compute the cos angle for c
//     [2.3] Find the evolution factor cevol(c)
//           and the Delta u(c) = du(c)
__host__
void QCS::setCornerDivGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double2> acc_ex(regions[0], FID_EXP);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    const AccessorRO<double2> acc_zx(regions[1], FID_ZXP);
    const AccessorRO<double2> acc_pu[2] = {
        AccessorRO<double2>(regions[2], FID_PU0),
        AccessorRO<double2>(regions[3], FID_PU0)
    };
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[2], FID_PXP),
        AccessorRO<double2>(regions[3], FID_PXP)
    };
    const AccessorWD<double2> acc_zuc(regions[4], FID_ZUC);
    const AccessorWD<double> acc_carea(regions[5], FID_CAREA);
    const AccessorWD<double> acc_ccos(regions[5], FID_CCOS);
    const AccessorWD<double> acc_cdiv(regions[5], FID_CDIV);
    const AccessorWD<double> acc_cevol(regions[5], FID_CEVOL);
    const AccessorWD<double> acc_cdu(regions[5], FID_CDU);

    // [1] Compute a zone-centered velocity
    const IndexSpace& isz = task->regions[1].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    cudaMemset(acc_zuc.ptr(rectz), 0, rectz.volume() * sizeof(double2));

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_zone_centered_velocity<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1,
        acc_mapsp1reg, acc_mapsz, acc_pu[0], acc_pu[1], acc_znump, acc_zuc,
        rects.lo, volume);

    // [2] Divergence at the corner
    gpu_corner_divergence<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsz, acc_mapsp1,
        acc_mapsp2, acc_mapss3, acc_mapsp1reg, acc_mapsp2reg, acc_ex, acc_elen,
        acc_znump, acc_zx, acc_pu[0], acc_pu[1], acc_px[0], acc_px[1], acc_zuc,
        acc_carea, acc_ccos, acc_cdiv, acc_cevol, acc_cdu, rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_compute_crmu(const AccessorRO<Pointer> acc_mapsz,
                 const AccessorRO<double> acc_cdu,
                 const AccessorRO<double> acc_zss,
                 const AccessorRO<double> acc_zrp,
                 const AccessorRO<double> acc_cevol,
                 const AccessorRO<double> acc_cdiv,
                 const AccessorWD<double> acc_crmu,
                 const double q1, const double q2, const double gammap1,
                 const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t c = origin[0] + offset;
  const Pointer z = acc_mapsz[c];

  // Kurapatenko form of the viscosity
  const double cdu = acc_cdu[c];
  const double ztmp2 = q2 * 0.25 * gammap1 * cdu;
  const double zss = acc_zss[z];
  const double ztmp1 = q1 * zss;
  const double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
  // Compute crmu for each corner
  const double zrp = acc_zrp[z];
  const double cevol = acc_cevol[c];
  const double crmu = zkur * zrp * cevol;
  const double cdiv = acc_cdiv[c];
  acc_crmu[c] = ((cdiv > 0.0) ? 0. : crmu);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_compute_cqe(const AccessorRO<Pointer> acc_mapsp1,
                const AccessorRO<Pointer> acc_mapsp2,
                const AccessorRO<Pointer> acc_mapss3,
                const AccessorRO<int> acc_mapsp1reg,
                const AccessorRO<int> acc_mapsp2reg,
                const AccessorRO<double2> acc_pu0,
                const AccessorRO<double2> acc_pu1,
                const AccessorRO<double> acc_elen,
                const AccessorWD<double> acc_crmu,
                const AccessorWD<double2> acc_cqe1,
                const AccessorWD<double2> acc_cqe2,
                const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t c = origin[0] + offset;
  const Pointer s2 = c;
  const Pointer s = acc_mapss3[s2];
  const Pointer p = acc_mapsp2[s];
  const int preg = acc_mapsp2reg[s];
  // Associated point 1
  const Pointer p1 = acc_mapsp1[s];
  const int p1reg = acc_mapsp1reg[s];
  // Associated point 2
  const Pointer p2 = acc_mapsp2[s2];
  const int p2reg = acc_mapsp2reg[s2];

  // Compute: cqe(1,2,3)=edge 1, y component (2nd), 3rd corner
  //          cqe(2,1,3)=edge 2, x component (1st)
  const double crmu = acc_crmu[c];
  const double2 pu = (preg == 0) ? acc_pu0[p] : acc_pu1[p];
  const double2 pu1 = (p1reg == 0) ? acc_pu0[p1] : acc_pu1[p1];
  const double elen = acc_elen[s];
  const double2 cqe1 = crmu * (pu - pu1) / elen;
  acc_cqe1[c] = cqe1;
  const double2 pu2 = (p2reg == 0) ? acc_pu0[p2] : acc_pu1[p2];
  const double elen2 = acc_elen[s2];
  const double2 cqe2 = crmu * (pu2 - pu) / elen2;
  acc_cqe2[c] = cqe2;
}

// Routine number [4]  in the full algorithm CS2DQforce(...)
__host__
void QCS::setQCnForceGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double qgamma = args[0];
    const double q1     = args[1];
    const double q2     = args[2];

    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<double> acc_cdiv(regions[0], FID_CDIV);
    const AccessorRO<double> acc_cdu(regions[0], FID_CDU);
    const AccessorRO<double> acc_cevol(regions[0], FID_CEVOL);
    const AccessorRO<double> acc_zrp(regions[1], FID_ZRP);
    const AccessorRO<double> acc_zss(regions[1], FID_ZSS);
    const AccessorRO<double2> acc_pu[2] = {
        AccessorRO<double2>(regions[2], FID_PU0),
        AccessorRO<double2>(regions[3], FID_PU0)
    };
    const AccessorWD<double> acc_crmu(regions[4], FID_CRMU);
    const AccessorWD<double2> acc_cqe1(regions[4], FID_CQE1);
    const AccessorWD<double2> acc_cqe2(regions[4], FID_CQE2);

    const double gammap1 = qgamma + 1.0;

    // [4.1] Compute the crmu (real Kurapatenko viscous scalar)
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_compute_crmu<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsz, acc_cdu, acc_zss, 
        acc_zrp, acc_cevol, acc_cdiv, acc_crmu, q1, q2, gammap1, rects.lo, volume);

    // [4.2] Compute the cqe for each corner
    gpu_compute_cqe<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1, acc_mapsp2,
        acc_mapss3, acc_mapsp1reg, acc_mapsp2reg, acc_pu[0], acc_pu[1],
        acc_elen, acc_crmu, acc_cqe1, acc_cqe2, rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_set_force1(const AccessorRW<double> acc_ccos,
               const AccessorRO<double> acc_carea,
               const AccessorWD<double> acc_cw,
               const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t c = origin[0] + offset;
  const double ccos = acc_ccos[c];
  const double csin2 = 1.0 - ccos * ccos;
  const double carea = acc_carea[c];
  const double cw = ((csin2 < 1.e-4) ? 0. : carea / csin2);
  acc_cw[c] = cw;
  acc_ccos[c] = ((csin2 < 1.e-4) ? 0. : ccos);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_set_force2(const AccessorRO<Pointer> acc_mapss4,
               const AccessorRO<double2> acc_cqe1,
               const AccessorRO<double2> acc_cqe2,
               const AccessorRO<double> acc_elen,
               const AccessorRW<double> acc_ccos,
               const AccessorWD<double> acc_cw,
               const AccessorWD<double2> acc_sfq,
               const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  // Associated corners 1 and 2
  const Pointer c1 = s;
  const Pointer c2 = acc_mapss4[s];
  // Edge length for c1, c2 contribution to s
  const double el = acc_elen[s];

  const double cw1 = acc_cw[c1];
  const double cw2 = acc_cw[c2];
  const double ccos1 = acc_ccos[c1];
  const double ccos2 = acc_ccos[c2];
  const double2 cqe11 = acc_cqe1[c1];
  const double2 cqe12 = acc_cqe1[c2];
  const double2 cqe21 = acc_cqe2[c1];
  const double2 cqe22 = acc_cqe2[c2];
  const double2 sfq = (cw1 * (cqe21 + ccos1 * cqe11) +
                 cw2 * (cqe12 + ccos2 * cqe22)) / el;
  acc_sfq[s] = sfq;
}

// Routine number [5]  in the full algorithm CS2DQforce(...)
__host__
void QCS::setForceGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapss4(regions[0], FID_MAPSS4);
    const AccessorRO<double> acc_carea(regions[0], FID_CAREA);
    const AccessorRO<double2> acc_cqe1(regions[0], FID_CQE1);
    const AccessorRO<double2> acc_cqe2(regions[0], FID_CQE2);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRW<double> acc_ccos(regions[1], FID_CCOS);
    const AccessorWD<double> acc_cw(regions[2], FID_CW);
    const AccessorWD<double2> acc_sfq(regions[2], FID_SFQ);

    // [5.1] Preparation of extra variables
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_set_force1<<<blocks,THREADS_PER_BLOCK>>>(acc_ccos, acc_carea,
                                                 acc_cw, rects.lo, volume);

    // [5.2] Set-Up the forces on corners
    gpu_set_force2<<<blocks,THREADS_PER_BLOCK>>>(acc_mapss4, acc_cqe1,
        acc_cqe2, acc_elen, acc_ccos, acc_cw, acc_sfq, rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_set_vel_diff1(const AccessorRO<Pointer> acc_mapsz,
                  const AccessorRO<Pointer> acc_mapsp1,
                  const AccessorRO<Pointer> acc_mapsp2,
                  const AccessorRO<int> acc_mapsp1reg,
                  const AccessorRO<int> acc_mapsp2reg,
                  const AccessorRO<double> acc_elen,
                  const AccessorRO<double2> acc_pu0,
                  const AccessorRO<double2> acc_pu1,
                  const AccessorRO<double2> acc_px0,
                  const AccessorRO<double2> acc_px1,
                  const AccessorWD<double> acc_ztmp,
                  const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer p1 = acc_mapsp1[s];
  const int p1reg = acc_mapsp1reg[s];
  const Pointer p2 = acc_mapsp2[s];
  const int p2reg = acc_mapsp2reg[s];
  const Pointer z = acc_mapsz[s];

  const double2 px1 = (p1reg == 0) ? acc_px0[p1] : acc_px1[p1];
  const double2 px2 = (p2reg == 0) ? acc_px0[p2] : acc_px1[p2];
  const double2 pu1 = (p1reg == 0) ? acc_pu0[p1] : acc_pu1[p1];
  const double2 pu2 = (p2reg == 0) ? acc_pu0[p2] : acc_pu1[p2];
  const double2 dx  = px2 - px1;
  const double2 du  = pu2 - pu1;
  const double lenx = acc_elen[s];
  double dux = dot(du, dx);
  dux = (lenx > 0. ? abs(dux) / lenx : 0.);

  MaxOp<double>::apply<false/*exclusive*/>(acc_ztmp[z], dux);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_set_vel_diff2(const AccessorRO<double> acc_zss,
                  const AccessorWD<double> acc_ztmp,
                  const AccessorWD<double> acc_zdu,
                  const double q1, const double q2,
                  const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t z = origin[0] + offset;
  const double zss  = acc_zss[z];
  const double ztmp  = acc_ztmp[z];
  const double zdu = q1 * zss + 2. * q2 * ztmp;
  acc_zdu[z] = zdu;
}

// Routine number [6] in the full algorithm
__host__
void QCS::setVelDiffGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double q1 = args[0];
    const double q2 = args[1];

    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<double> acc_zss(regions[1], FID_ZSS);
    const AccessorRO<double2> acc_pu[2] = {
        AccessorRO<double2>(regions[2], FID_PU0),
        AccessorRO<double2>(regions[3], FID_PU0)
    };
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[2], FID_PXP),
        AccessorRO<double2>(regions[3], FID_PXP)
    };
    const AccessorWD<double> acc_ztmp(regions[4], FID_ZTMP);
    const AccessorWD<double> acc_zdu(regions[4], FID_ZDU);

    const IndexSpace& isz = task->regions[4].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    const size_t volumez = rectz.volume();
    cudaMemset(acc_ztmp.ptr(rectz), 0, volumez * sizeof(double));

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_set_vel_diff1<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsz, acc_mapsp1,
        acc_mapsp2, acc_mapsp1reg, acc_mapsp2reg, acc_elen, acc_pu[0], acc_pu[1],
        acc_px[0], acc_px[1], acc_ztmp, rects.lo, volume);

    const size_t blockz = (volumez + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_set_vel_diff2<<<blockz,THREADS_PER_BLOCK>>>(acc_zss, acc_ztmp,
        acc_zdu, q1, q2, rectz.lo, volumez);
}

