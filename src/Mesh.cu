
#include "Mesh.hh"
#include "MyLegion.hh"
#include "CudaHelp.hh"

using namespace Legion;

namespace {  // unnamed
__host__
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_CALCCTRS, "GPU calcctrs");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcCtrsGPUTask>(registrar, "calcctrs");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCVOLS, "GPU calcvols");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<DeferredReduction<SumOp<int> >, 
        Mesh::calcVolsGPUTask>(registrar, "calcvols");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCEDGELEN, "GPU calcedgelen");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcEdgeLenGPUTask>(registrar, "calcedgelen");
    }
}
}; // namespace

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_ctrs(const AccessorRO<Pointer> acc_mapsp1,
              const AccessorRO<Pointer> acc_mapsp2,
              const AccessorRO<Pointer> acc_mapsz,
              const AccessorRO<int> acc_mapsp1reg,
              const AccessorRO<int> acc_mapsp2reg,
              const AccessorRO<int> acc_znump,
              const AccessorRO<double2> acc_px0,
              const AccessorRO<double2> acc_px1,
              const AccessorWD<double2> acc_ex,
              const AccessorWD<double2> acc_zx,
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
  const double2 ex  = 0.5 * (px1 + px2);
  acc_ex[s] = ex;
  const int n = acc_znump[z];
  SumOp<double2>::apply<false/*exclusive*/>(acc_zx[z], px1 / n);
}

__host__
void Mesh::calcCtrsGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    FieldID fid_px = task->regions[2].instance_fields[0];
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[2], fid_px),
        AccessorRO<double2>(regions[3], fid_px)
    };
    FieldID fid_ex = task->regions[4].instance_fields[0];
    const AccessorWD<double2> acc_ex(regions[4], fid_ex);
    FieldID fid_zx = task->regions[5].instance_fields[0];
    const AccessorWD<double2> acc_zx(regions[5], fid_zx);

    const IndexSpace& isz = task->regions[1].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    cudaMemset(acc_zx.ptr(rectz), 0, rectz.volume() * sizeof(double2));

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_ctrs<<<blocks,volume>>>(acc_mapsp1, acc_mapsp2, acc_mapsz,
        acc_mapsp1reg, acc_mapsp2reg, acc_znump, acc_px[0], acc_px[1],
        acc_ex, acc_zx, rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_vols(const AccessorRO<Pointer> acc_mapsp1,
              const AccessorRO<Pointer> acc_mapsp2,
              const AccessorRO<Pointer> acc_mapsz,
              const AccessorRO<int> acc_mapsp1reg,
              const AccessorRO<int> acc_mapsp2reg,
              const AccessorRO<double2> acc_px0,
              const AccessorRO<double2> acc_px1,
              const AccessorRO<double2> acc_zx,
              const AccessorWD<double> acc_sarea,
              const AccessorWD<double> acc_svol,
              const AccessorWD<double> acc_zarea,
              const AccessorWD<double> acc_zvol,
              DeferredReduction<SumOp<int> > result,
              const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const double third = 1. / 3.;
  const Pointer p1 = acc_mapsp1[s];
  const int p1reg = acc_mapsp1reg[s];
  const Pointer p2 = acc_mapsp2[s];
  const int p2reg = acc_mapsp2reg[s];
  const Pointer z = acc_mapsz[s];
  const double2 px1 = (p1reg == 0) ? acc_px0[p1] : acc_px1[p1];
  const double2 px2 = (p2reg == 0) ? acc_px0[p2] : acc_px1[p2];
  const double2 zx  = acc_zx[z];

  // compute side volumes, sum to zone
  const double sa = 0.5 * cross(px2 - px1, zx - px1);
  const double sv = third * sa * (px1.x + px2.x + zx.x);
  acc_sarea[s] = sa;
  acc_svol[s] = sv;
  SumOp<double>::apply<false/*exclusive*/>(acc_zarea[z], sa);
  SumOp<double>::apply<false/*exclusive*/>(acc_zvol[z], sv);

  // check for negative side volumes
  if (sv <= 0.) 
    result <<= 1;
}

__host__
DeferredReduction<SumOp<int> > Mesh::calcVolsGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    FieldID fid_px = task->regions[1].instance_fields[0];
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[1], fid_px),
        AccessorRO<double2>(regions[2], fid_px)
    };
    FieldID fid_zx = task->regions[3].instance_fields[0];
    const AccessorRO<double2> acc_zx(regions[3], fid_zx);
    FieldID fid_sarea = task->regions[4].instance_fields[0];
    FieldID fid_svol  = task->regions[4].instance_fields[1];
    const AccessorWD<double> acc_sarea(regions[4], fid_sarea);
    const AccessorWD<double> acc_svol(regions[4], fid_svol);
    FieldID fid_zarea = task->regions[5].instance_fields[0];
    FieldID fid_zvol  = task->regions[5].instance_fields[1];
    const AccessorWD<double> acc_zarea(regions[5], fid_zarea);
    const AccessorWD<double> acc_zvol(regions[5], fid_zvol);

    const IndexSpace& isz = task->regions[3].region.get_index_space();
    // This will assert if it isn't dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    cudaMemset(acc_zarea.ptr(rectz), 0, rectz.volume() * sizeof(double));
    cudaMemset(acc_zvol.ptr(rectz), 0, rectz.volume() * sizeof(double));

    DeferredReduction<SumOp<int> > result;

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it isn't dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_vols<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1, acc_mapsp2,
        acc_mapsz, acc_mapsp1reg, acc_mapsp2reg, acc_px[0], acc_px[1],
        acc_zx, acc_sarea, acc_svol, acc_zarea, acc_zvol, result,
        rects.lo, volume);
    return result;
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_edge_length(const AccessorRO<Pointer> acc_mapsp1,
                     const AccessorRO<Pointer> acc_mapsp2,
                     const AccessorRO<int> acc_mapsp1reg,
                     const AccessorRO<int> acc_mapsp2reg,
                     const AccessorRO<double2> acc_px0,
                     const AccessorRO<double2> acc_px1,
                     const AccessorWD<double> acc_elen,
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
  const double2 px1 = (p1reg == 0) ? acc_px0[p1] : acc_px1[p1];
  const double2 px2 = (p2reg == 0) ? acc_px0[p2] : acc_px1[p2];

  const double elen = length(px2 - px1);
  acc_elen[s] = elen;
}

__host__
void Mesh::calcEdgeLenGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double2> acc_px[2] = {
        AccessorRO<double2>(regions[1], FID_PXP),
        AccessorRO<double2>(regions[2], FID_PXP)
    };
    const AccessorWD<double> acc_elen(regions[3], FID_ELEN);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_edge_length<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1, acc_mapsp2,
        acc_mapsp1reg, acc_mapsp2reg, acc_px[0], acc_px[1], acc_elen,
        rects.lo, volume);
}

