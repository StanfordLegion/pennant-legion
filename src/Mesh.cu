
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
      add_colocation_constraint(registrar, 2, 3, FID_PXP);
      Runtime::preregister_task_variant<Mesh::calcCtrsGPUTask>(registrar, "calcctrs");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCVOLS, "GPU calcvols");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      add_colocation_constraint(registrar, 1, 2, FID_PXP);
      Runtime::preregister_task_variant<DeferredReduction<SumOp<int> >, 
        Mesh::calcVolsGPUTask>(registrar, "calcvols");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCEDGELEN, "GPU calcedgelen");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcEdgeLenGPUTask>(registrar, "calcedgelen");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCSURFVECS, "GPU calcsurfvecs");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcSurfVecsGPUTask>(registrar, "calcsurfvecs");
    }
    {
      TaskVariantRegistrar registrar(TID_CALCCHARLEN, "GPU calccharlen");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<Mesh::calcCharLenGPUTask>(registrar, "calccharlen");
    }
}
}; // namespace

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_ctrs(const AccessorRO<Pointer> acc_mapsp1,
              const AccessorRO<Pointer> acc_mapsp2,
              const AccessorRO<Pointer> acc_mapsz,
              const AccessorRO<int> acc_znump,
              const AccessorMC<double2> acc_px,
              const AccessorWD<double2> acc_ex,
              const AccessorWD<double2> acc_zx,
              const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer p1 = acc_mapsp1[s];
  const Pointer p2 = acc_mapsp2[s];
  const Pointer z = acc_mapsz[s];
  const double2 px1 = acc_px[p1];
  const double2 px2 = acc_px[p2];
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
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    FieldID fid_px = task->regions[2].instance_fields[0];
    const AccessorMC<double2> acc_px(regions.begin()+2, regions.begin()+4, fid_px);
    FieldID fid_ex = task->regions[4].instance_fields[0];
    const AccessorWD<double2> acc_ex(regions[4], fid_ex);
    FieldID fid_zx = task->regions[5].instance_fields[0];
    const AccessorWD<double2> acc_zx(regions[5], fid_zx);

    const IndexSpace& isz = task->regions[1].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    if (rectz.empty())
      return;
    cudaMemset(acc_zx.ptr(rectz), 0, rectz.volume() * sizeof(double2));

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_ctrs<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1, acc_mapsp2, acc_mapsz,
        acc_znump, acc_px, acc_ex, acc_zx, rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_vols(const AccessorRO<Pointer> acc_mapsp1,
              const AccessorRO<Pointer> acc_mapsp2,
              const AccessorRO<Pointer> acc_mapsz,
              const AccessorMC<double2> acc_px,
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
  const Pointer p2 = acc_mapsp2[s];
  const Pointer z = acc_mapsz[s];
  const double2 px1 = acc_px[p1];
  const double2 px2 = acc_px[p2];
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
    FieldID fid_px = task->regions[1].instance_fields[0];
    const AccessorMC<double2> acc_px(regions.begin()+1, regions.begin()+3, fid_px);
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

    DeferredReduction<SumOp<int> > result;

    const IndexSpace& isz = task->regions[3].region.get_index_space();
    // This will assert if it isn't dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    if (rectz.empty())
      return result;
    cudaMemset(acc_zarea.ptr(rectz), 0, rectz.volume() * sizeof(double));
    cudaMemset(acc_zvol.ptr(rectz), 0, rectz.volume() * sizeof(double));

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it isn't dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return result;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_vols<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1, acc_mapsp2,
        acc_mapsz, acc_px, acc_zx, acc_sarea, acc_svol, acc_zarea, 
        acc_zvol, result, rects.lo, volume);
    return result;
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_edge_length(const AccessorRO<Pointer> acc_mapsp1,
                     const AccessorRO<Pointer> acc_mapsp2,
                     const AccessorMC<double2> acc_px,
                     const AccessorWD<double> acc_elen,
                     const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer p1 = acc_mapsp1[s];
  const Pointer p2 = acc_mapsp2[s];
  const double2 px1 = acc_px[p1];
  const double2 px2 = acc_px[p2];

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
    const AccessorMC<double2> acc_px(regions.begin()+1, regions.begin()+3, FID_PXP);
    const AccessorWD<double> acc_elen(regions[3], FID_ELEN);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_edge_length<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsp1, acc_mapsp2,
        acc_px, acc_elen, rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_surf_vecs(const AccessorRO<Pointer> acc_mapsz,
                   const AccessorRO<double2> acc_ex,
                   const AccessorRO<double2> acc_zx,
                   const AccessorWD<double2> acc_ssurf,
                   const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer z = acc_mapsz[s];
  const double2 ex = acc_ex[s];
  const double2 zx = acc_zx[z];
  const double2 ss = rotateCCW(ex - zx);
  acc_ssurf[s] = ss;
}

__host__
void Mesh::calcSurfVecsGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double2> acc_ex(regions[0], FID_EXP);
    const AccessorRO<double2> acc_zx(regions[1], FID_ZXP);
    const AccessorWD<double2> acc_ssurf(regions[2], FID_SSURFP);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_surf_vecs<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsz, acc_ex,
        acc_zx, acc_ssurf, rects.lo, volume);
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_char_len_init(const AccessorWD<double> acc_zdl, const double value,
                       const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t z = origin[0] + offset;   
  acc_zdl[z] = value;
}

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_calc_char_len(const AccessorRO<Pointer> acc_mapsz,
                  const AccessorRO<double> acc_elen,
                  const AccessorRO<double> acc_sarea,
                  const AccessorRO<int> acc_znump,
                  const AccessorWD<double> acc_zdl,
                  const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t s = origin[0] + offset;
  const Pointer z = acc_mapsz[s];
  const double area = acc_sarea[s];
  const double base = acc_elen[s];
  const int np = acc_znump[z];
  const double fac = (np == 3 ? 3. : 4.);
  const double sdl = fac * area / base;
  MinOp<double>::apply<false/*exclusive*/>(acc_zdl[z], sdl);
}

__host__
void Mesh::calcCharLenGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<double> acc_sarea(regions[0], FID_SAREAP);
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    const AccessorWD<double> acc_zdl(regions[2], FID_ZDL);

    const IndexSpace& isz = task->regions[1].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    const size_t volumez = rectz.volume();
    if (volumez == 0)
      return;
    const size_t blockz = (volumez + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_char_len_init<<<blockz,THREADS_PER_BLOCK>>>(acc_zdl, 1.e99, 
        rectz.lo, volumez);
    
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    const size_t volume = rects.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_calc_char_len<<<blocks,THREADS_PER_BLOCK>>>(acc_mapsz, acc_elen,
        acc_sarea, acc_znump, acc_zdl, rects.lo, volume);
}
