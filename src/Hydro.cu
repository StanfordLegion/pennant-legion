
#include "Hydro.hh"
#include "MyLegion.hh"
#include "CudaHelp.hh"

using namespace Legion;

namespace {  // unnamed
__host__
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_ADVPOSHALF, "GPU advposhalf");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::advPosHalfGPUTask>(registrar, "advposhalf");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCRHO, "GPU calcrho");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcRhoGPUTask>(registrar, "calcrho");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCCRNRMASS, "GPU calccrnrmass");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcCrnrMassGPUTask>(registrar, "calccrnrmass");
    }
    {
      TaskVariantRegistrar registrar(TID_SUMCRNRFORCE, "GPU sumcrnrforce");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::sumCrnrForceGPUTask>(registrar, "sumcrnrforce");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCACCEL, "GPU calcaccel");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcAccelGPUTask>(registrar, "calcaccel"); 
    }
    {
      TaskVariantRegistrar registrar(TID_ADVPOSFULL, "GPU advposfull");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::advPosFullGPUTask>(registrar, "advposfull");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCWORK, "GPU calcwork");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcWorkGPUTask>(registrar, "calcwork");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCWORKRATE, "GPU calcworkrate");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcWorkRateGPUTask>(registrar, "calcworkrate");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCENERGY, "GPU calcenergy");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Hydro::calcEnergyGPUTask>(registrar, "calcenergy");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCDTNEW, "GPU calcdtnew");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<DeferredReduction<MinOp<double> >, 
        Hydro::calcDtNewGPUTask>(registrar, "calcdtnew");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCDVOL, "GPU calcdvol");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<DeferredReduction<MaxOp<double> >, 
        Hydro::calcDvolGPUTask>(registrar, "calcdvol");
    }
}
}; // namespace

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_adv_pos_half(const AccessorRO<double2> acc_px0,
                 const AccessorRO<double2> acc_pu0,
                 const AccessorWD<double2> acc_pxp,
                 const Point<1> origin,
                 const double dth, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t p = origin[0] + offset;
  const double2 x0 = acc_px0[p];
  const double2 u0 = acc_pu0[p];
  const double2 xp = x0 + dth * u0;
  acc_pxp[p] = xp;
}

__host__
void Hydro::advPosHalfGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();
    const double dth = 0.5 * dt;
    const AccessorRO<double2> acc_px0(regions[0], FID_PX0);
    const AccessorRO<double2> acc_pu0(regions[0], FID_PU0);
    const AccessorWD<double2> acc_pxp(regions[1], FID_PXP);

    const IndexSpace& isp = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectp = runtime->get_index_space_domain(isp);
    const size_t volume = rectp.volume();
    // Points can be empty in some cases
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_adv_pos_half<<<blocks,THREADS_PER_BLOCK>>>(acc_px0, acc_pu0, acc_pxp,
                                                   rectp.lo, dth, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_rho(const AccessorRO<double> acc_zm,
             const AccessorRO<double> acc_zvol,
             const AccessorWD<double> acc_zr,
             const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t z = origin[0] + offset;
  const double m = acc_zm[z];
  const double v = acc_zvol[z];
  const double r = m / v;
  acc_zr[z] = r;
}

__host__
void Hydro::calcRhoGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    FieldID fid_zm = task->regions[0].instance_fields[0];
    FieldID fid_zvol = task->regions[0].instance_fields[1];
    const AccessorRO<double> acc_zm(regions[0], fid_zm);
    const AccessorRO<double> acc_zvol(regions[0], fid_zvol);
    FieldID fid_zr = task->regions[1].instance_fields[0];
    const AccessorWD<double> acc_zr(regions[1], fid_zr);

    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    const size_t volume = rectz.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_rho<<<blocks,THREADS_PER_BLOCK>>>(acc_zm, acc_zvol, acc_zr,
                                               rectz.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_crnr_mass(const AccessorRO<Pointer> acc_mapsp1,
                   const AccessorRO<int> acc_mapsp1reg,
                   const AccessorRO<Pointer> acc_mapss3,
                   const AccessorRO<Pointer> acc_mapsz,
                   const AccessorRO<double> acc_smf,
                   const AccessorRO<double> acc_zr,
                   const AccessorRO<double> acc_zarea,
                   const AccessorRW<double> acc_pmas_prv,
                   const AccessorRD<SumOp<double>,false/*exclusive*/> acc_pmas_shr,
                   const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer s3 = acc_mapss3[s];
  const Pointer z  = acc_mapsz[s];
  const Pointer p = acc_mapsp1[s];
  const int preg = acc_mapsp1reg[s];
  const double r = acc_zr[z];
  const double area = acc_zarea[z];
  const double mf = acc_smf[s];
  const double mf3 = acc_smf[s3];
  const double mwt = r * area * 0.5 * (mf + mf3);
  if (preg == 0)
      SumOp<double>::apply<false/*exclusive*/>(acc_pmas_prv[p], mwt);
  else
      acc_pmas_shr[p] <<= mwt;
}

__host__
void Hydro::calcCrnrMassGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double> acc_smf(regions[0], FID_SMF);
    const AccessorRO<double> acc_zr(regions[1], FID_ZRP);
    const AccessorRO<double> acc_zarea(regions[1], FID_ZAREAP);
    const AccessorRW<double> acc_pmas_prv(regions[2], FID_PMASWT);
    const AccessorRD<SumOp<double>,false/*exclusive*/> 
      acc_pmas_shr(regions[3], FID_PMASWT, OPID_SUMDBL);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_crnr_mass<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1, acc_mapsp1reg,
        acc_mapss3, acc_mapsz, acc_smf, acc_zr, acc_zarea, acc_pmas_prv,
        acc_pmas_shr, rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_sum_crnr_force(const AccessorRO<Pointer> acc_mapsp1,
                   const AccessorRO<int> acc_mapsp1reg,
                   const AccessorRO<Pointer> acc_mapss3,
                   const AccessorRO<double2> acc_sfp,
                   const AccessorRO<double2> acc_sfq,
                   const AccessorRO<double2> acc_sft,
                   const AccessorRW<double2> acc_pf_prv,
                   const AccessorRD<SumOp<double2>,false/*exclusive*/> acc_pf_shr,
                   const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer s3 = acc_mapss3[s];
  const Pointer p = acc_mapsp1[s];
  const int preg = acc_mapsp1reg[s];
  const double2 sfp = acc_sfp[s];
  const double2 sfq = acc_sfq[s];
  const double2 sft = acc_sft[s];
  const double2 sfp3 = acc_sfp[s3];
  const double2 sfq3 = acc_sfq[s3];
  const double2 sft3 = acc_sft[s3];
  const double2 cf = (sfp + sfq + sft) - (sfp3 + sfq3 + sft3);
  if (preg == 0)
      SumOp<double2>::apply<false/*exclusive*/>(acc_pf_prv[p], cf);
  else
      acc_pf_shr[p] <<= cf;
}

__host__
void Hydro::sumCrnrForceGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<double2> acc_sfp(regions[0], FID_SFP);
    const AccessorRO<double2> acc_sfq(regions[0], FID_SFQ);
    const AccessorRO<double2> acc_sft(regions[0], FID_SFT);
    const AccessorRW<double2> acc_pf_prv(regions[1], FID_PF);
    const AccessorRD<SumOp<double2>,false/*exclusive*/> 
      acc_pf_shr(regions[2], FID_PF, OPID_SUMDBL2);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_sum_crnr_force<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1, acc_mapsp1reg,
        acc_mapss3, acc_sfp, acc_sfq, acc_sft, acc_pf_prv, acc_pf_shr, rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_accel(const AccessorRO<double2> acc_pf,
               const AccessorRO<double> acc_pmass,
               const AccessorWD<double2> acc_pa,
               const double fuzz, const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t p = origin[0] + offset;
  const double2 f = acc_pf[p];
  const double m = acc_pmass[p];
  const double2 a = f / ((m > fuzz) ? m : fuzz);
  acc_pa[p] = a;
}

__host__
void Hydro::calcAccelGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<double2> acc_pf(regions[0], FID_PF);
    const AccessorRO<double> acc_pmass(regions[0], FID_PMASWT);
    const AccessorWD<double2> acc_pa(regions[1], FID_PAP);

    const double fuzz = 1.e-99;
    const IndexSpace& isp = task->regions[0].region.get_index_space();
    // This will assert if its not dense
    const Rect<1> rectp = runtime->get_index_space_domain(isp);
    const size_t volume = rectp.volume();
    // Points can be empty in some cases
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_accel<<<blocks,THREADS_PER_BLOCK>>>(acc_pf, acc_pmass, acc_pa, 
                                                 fuzz, rectp.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_adv_pos_full(const AccessorRO<double2> acc_px0,
                 const AccessorRO<double2> acc_pu0,
                 const AccessorRO<double2> acc_pa,
                 const AccessorWD<double2> acc_px,
                 const AccessorWD<double2> acc_pu,
                 const double dt, const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t p = origin[0] + offset;
  const double2 x0 = acc_px0[p];
  const double2 u0 = acc_pu0[p];
  const double2 a = acc_pa[p];
  const double2 u = u0 + dt * a;
  acc_pu[p] = u;
  const double2 x = x0 + dt * 0.5 * (u0 + u);
  acc_px[p] = x;
}

__host__
void Hydro::advPosFullGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();

    const AccessorRO<double2> acc_px0(regions[0], FID_PX0);
    const AccessorRO<double2> acc_pu0(regions[0], FID_PU0);
    const AccessorRO<double2> acc_pa(regions[0], FID_PAP);
    const AccessorWD<double2> acc_px(regions[1], FID_PX);
    const AccessorWD<double2> acc_pu(regions[1], FID_PU);

    const IndexSpace& isp = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectp = runtime->get_index_space_domain(isp);
    const size_t volume = rectp.volume();
    // Points can be empty in some cases
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_adv_pos_full<<<blocks,THREADS_PER_BLOCK>>>(acc_px0, acc_pu0, acc_pa,
        acc_px, acc_pu, dt, rectp.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_work(const AccessorRO<Pointer> acc_mapsp1,
              const AccessorRO<Pointer> acc_mapsp2,
              const AccessorRO<Pointer> acc_mapsz,
              const AccessorRO<int> acc_mapsp1reg,
              const AccessorRO<int> acc_mapsp2reg,
              const AccessorRO<double2> acc_sf,
              const AccessorRO<double2> acc_sf2,
              const AccessorRO<double2> acc_pu00,
              const AccessorRO<double2> acc_pu01,
              const AccessorRO<double2> acc_pu0,
              const AccessorRO<double2> acc_pu1,
              const AccessorRO<double2> acc_px0,
              const AccessorRO<double2> acc_px1,
              const AccessorRW<double> acc_zw,
              const AccessorRW<double> acc_zetot,
              const double dth, const Point<1> origin, const size_t max)
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
  const double2 sf = acc_sf[s];
  const double2 sf2 = acc_sf2[s];
  const double2 sftot = sf + sf2;
  const double2 pu01 = (p1reg == 0) ? acc_pu00[p1] : acc_pu01[p1];
  const double2 pu1 = (p1reg == 0) ? acc_pu0[p1] : acc_pu1[p1];
  const double sd1 = dot(sftot, (pu01 + pu1));
  const double2 pu02 = (p2reg == 0) ? acc_pu00[p2] : acc_pu01[p2];
  const double2 pu2 = (p2reg == 0) ? acc_pu0[p2] : acc_pu1[p2];
  const double sd2 = dot(-sftot, (pu02 + pu2));
  const double2 px1 = (p1reg == 0) ? acc_px0[p1] : acc_px1[p1];
  const double2 px2 = (p2reg == 0) ? acc_px0[p2] : acc_px1[p2];;
  const double dwork = -dth * (sd1 * px1.x + sd2 * px2.x);

  SumOp<double>::apply<false/*exclusive*/>(acc_zetot[z], dwork);
  SumOp<double>::apply<false/*exclusive*/>(acc_zw[z], dwork);
}

__host__
void Hydro::calcWorkGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();

    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double2> acc_sf(regions[0], FID_SFP);
    const AccessorRO<double2> acc_sf2(regions[0], FID_SFQ);
    const AccessorRO<double2> acc_pu0[2] = {
        AccessorRO<double2>(regions[1], FID_PU0),
        AccessorRO<double2>(regions[2], FID_PU0)
    };
    const AccessorRO<double2> acc_pu[2] = {
        AccessorRO<double2>(regions[1], FID_PU),
        AccessorRO<double2>(regions[2], FID_PU)
    };
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[1], FID_PXP),
        AccessorRO<double2>(regions[2], FID_PXP)
    };
    const AccessorRW<double> acc_zw(regions[3], FID_ZW);
    const AccessorRW<double> acc_zetot(regions[4], FID_ZETOT);

    // Compute the work done by finding, for each element/node pair,
    //   dwork= force * vavg
    // where force is the force of the element on the node
    // and vavg is the average velocity of the node over the time period

    const IndexSpace& isz = task->regions[3].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    cudaMemset(acc_zw.ptr(rectz), 0, rectz.volume() * sizeof(double));

    const double dth = 0.5 * dt;

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss); 
    const size_t volume = rects.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_work<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1, acc_mapsp2, acc_mapsz,
        acc_mapsp1reg, acc_mapsp2reg, acc_sf, acc_sf2, acc_pu0[0], acc_pu0[1],
        acc_pu[0], acc_pu[1], acc_px[0], acc_px[1], acc_zw, acc_zetot, dth,
        rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_work_rate(const AccessorRO<double> acc_zvol0,
                   const AccessorRO<double> acc_zvol,
                   const AccessorRO<double> acc_zw,
                   const AccessorRO<double> acc_zp,
                   const AccessorWD<double> acc_zwrate,
                   const double dtinv, const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t z = origin[0] + offset;
  const double zvol = acc_zvol[z];
  const double zvol0 = acc_zvol0[z];
  const double dvol = zvol - zvol0;
  const double zw = acc_zw[z];
  const double zp = acc_zp[z];
  const double zwrate = (zw + zp * dvol) * dtinv;
  acc_zwrate[z] = zwrate;
}

__host__
void Hydro::calcWorkRateGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double dt = task->futures[0].get_result<double>();

    const AccessorRO<double> acc_zvol0(regions[0], FID_ZVOL0);
    const AccessorRO<double> acc_zvol(regions[0], FID_ZVOL);
    const AccessorRO<double> acc_zw(regions[0], FID_ZW);
    const AccessorRO<double> acc_zp(regions[0], FID_ZP);
    const AccessorWD<double> acc_zwrate(regions[1], FID_ZWRATE);

    double dtinv = 1. / dt;

    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    const size_t volume = rectz.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_work_rate<<<blocks,THREADS_PER_BLOCK>>>(acc_zvol0, acc_zvol,
        acc_zw, acc_zp, acc_zwrate, dtinv, rectz.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_energy(const AccessorRO<double> acc_zetot,
                const AccessorRO<double> acc_zm,
                const AccessorWD<double> acc_ze,
                const double fuzz, const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t z = origin[0] + offset;
  const double zetot = acc_zetot[z];
  const double zm = acc_zm[z];
  const double ze = zetot / (zm + fuzz);
  acc_ze[z] = ze;
}

__host__
void Hydro::calcEnergyGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<double> acc_zetot(regions[0], FID_ZETOT);
    const AccessorRO<double> acc_zm(regions[0], FID_ZM);
    const AccessorWD<double> acc_ze(regions[1], FID_ZE);

    const double fuzz = 1.e-99;
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    const size_t volume = rectz.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_energy<<<blocks,THREADS_PER_BLOCK>>>(acc_zetot, acc_zm, acc_ze,
                                                  fuzz, rectz.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_dt_new(const AccessorRO<double> acc_zdl,
                const AccessorRO<double> acc_zdu,
                const AccessorRO<double> acc_zss,
                DeferredReduction<MinOp<double> > result,
                const double cfl, const double fuzz,
                const size_t iters, const Point<1> origin, 
                const size_t max, const double identity)
{
  double dtnew = identity;
  for (unsigned idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < max) {
      const coord_t z = origin[0] + offset;
      const double zdu = acc_zdu[z];
      const double zss = acc_zss[z];
      const double pin = (zss > fuzz) ? zss : fuzz;
      const double cdu = (zdu > pin) ? zdu : pin;
      const double zdl = acc_zdl[z];
      const double zdthyd = zdl * cfl / cdu;
      dtnew = (zdthyd < dtnew ? zdthyd : dtnew);
    }
  }
  reduce_double<MinOp<double> >(result, dtnew);
}

// forward declaration to remove compiler warning
template<>
const double MinOp<double>::identity;

__host__
DeferredReduction<MinOp<double> > Hydro::calcDtNewGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double cfl    = args[0];

    const AccessorRO<double> acc_zdl(regions[0], FID_ZDL);
    const AccessorRO<double> acc_zdu(regions[0], FID_ZDU);
    const AccessorRO<double> acc_zss(regions[0], FID_ZSS);

    // compute dt using Courant condition
    const double fuzz = 1.e-99;
    DeferredReduction<MinOp<double> > dtnew;
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    const size_t volume = rectz.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      gpu_calc_dt_new<<<MAX_REDUCTION_CTAS,THREADS_PER_BLOCK>>>(acc_zdl, acc_zdu, 
          acc_zss, dtnew, cfl, fuzz, iters, rectz.lo, volume, MinOp<double>::identity);
    } else {
      gpu_calc_dt_new<<<blocks,THREADS_PER_BLOCK>>>(acc_zdl, acc_zdu, acc_zss, 
          dtnew, cfl, fuzz, 1/*iters*/, rectz.lo, volume, MinOp<double>::identity);
    }
    return dtnew;
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_dvol(const AccessorRO<double> acc_zvol,
              const AccessorRO<double> acc_zvol0,
              DeferredReduction<MaxOp<double> > result,
              const size_t iters, const Point<1> origin, 
              const size_t max, const double identity)
{
  double dvovmax = identity;
  for (unsigned idx = 0; idx < iters; idx++) {
    const size_t offset = (idx * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset < max) {
      const coord_t z = origin[0] + offset;
      const double zvol = acc_zvol[z];
      const double zvol0 = acc_zvol0[z];
      const double zdvov = abs((zvol - zvol0) / zvol0);
      dvovmax = (zdvov > dvovmax ? zdvov : dvovmax);
    }
  }
  reduce_double<MaxOp<double> >(result, dvovmax);
}

// forward declaration to remove compiler warning
template<>
const double MaxOp<double>::identity;

__host__
DeferredReduction<MaxOp<double> > Hydro::calcDvolGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {

    const AccessorRO<double> acc_zvol(regions[0], FID_ZVOL);
    const AccessorRO<double> acc_zvol0(regions[0], FID_ZVOL0);

    DeferredReduction<MaxOp<double> > dvol;
    const IndexSpace& isz = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    const size_t volume = rectz.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks >= MAX_REDUCTION_CTAS) {
      const size_t iters = (blocks + MAX_REDUCTION_CTAS - 1) / MAX_REDUCTION_CTAS;
      gpu_calc_dvol<<<MAX_REDUCTION_CTAS,THREADS_PER_BLOCK>>>(acc_zvol, acc_zvol0,
          dvol, iters, rectz.lo, volume, MaxOp<double>::identity);
    } else {
      gpu_calc_dvol<<<blocks,THREADS_PER_BLOCK>>>(acc_zvol, acc_zvol0,
          dvol, 1/*iters*/, rectz.lo, volume, MaxOp<double>::identity);
    }
    return dvol;
}

