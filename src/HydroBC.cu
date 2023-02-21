
#include "Mesh.hh"
#include "Hydro.hh"
#include "HydroBC.hh"
#include "MyLegion.hh"
#include "CudaHelp.hh"

using namespace ResilientLegion;

namespace {  // unnamed
__host__
static void __attribute__ ((constructor)) registerTasks() {
  {
    TaskVariantRegistrar registrar(TID_APPLYFIXEDBC, "GPU applyfixedbc");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<HydroBC::applyFixedBCGPUTask>(registrar, "applyfixedbc");
  }
}
}; // namespace

__global__ void
__launch_bounds__(THREADS_PER_BLOCK,MIN_CTAS_PER_SM)
gpu_apply_fixed_bc(const AccessorRO<Pointer> acc_mapbp,
                   const AccessorRO<int> acc_mapbpreg,
                   const AccessorRW<double2> acc_pf0,
                   const AccessorRW<double2> acc_pf1,
                   const AccessorRW<double2> acc_pu0,
                   const AccessorRW<double2> acc_pu1,
                   const double2 vfix,
                   const Point<1> origin, const size_t max)
{
  const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= max)
    return;
  const coord_t b = origin[0] + offset; 
  const Pointer p = acc_mapbp[b];
  const int preg = acc_mapbpreg[b];
  double2 pu = (preg == 0) ? acc_pu0[p] : acc_pu1[p];
  double2 pf = (preg == 0) ? acc_pf0[p] : acc_pf1[p];
  pu = project(pu, vfix);
  pf = project(pf, vfix);
  if (preg == 0) {
    acc_pu0[p] = pu;
    acc_pf0[p] = pf;
  } else {
    acc_pu1[p] = pu;
    acc_pf1[p] = pf;
  }
}

__host__
void HydroBC::applyFixedBCGPUTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double2* args = (const double2*) task->args;
    const double2 vfix = args[0];
    const AccessorRO<Pointer> acc_mapbp(regions[0], FID_MAPBP);
    const AccessorRO<int> acc_mapbpreg(regions[0], FID_MAPBPREG);
    const AccessorRW<double2> acc_pf[2] = {
        AccessorRW<double2>(regions[1], FID_PF),
        AccessorRW<double2>(regions[2], FID_PF)
    };
    const AccessorRW<double2> acc_pu[2] = {
        AccessorRW<double2>(regions[1], FID_PU0),
        AccessorRW<double2>(regions[2], FID_PU0)
    };

    const IndexSpace& isb = task->regions[0].region.get_index_space();
    // This will fail if it is not dense
    const Rect<1> rectb = runtime->get_index_space_domain(isb);
    const size_t volume = rectb.volume();
    if (volume == 0)
      return;
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    gpu_apply_fixed_bc<<<blocks,THREADS_PER_BLOCK>>>(acc_mapbp, acc_mapbpreg,
        acc_pf[0], acc_pf[1], acc_pu[0], acc_pu[1], vfix, rectb.lo, volume);
}

