/*
 * Mesh.hh
 *
 *  Created on: Jan 5, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef MESH_HH_
#define MESH_HH_

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "resilience.h"
#include "default_mapper.h"

#include "Vec2.hh"
#include "GenMesh.hh"
#include "CudaHelp.hh"

#define Legion ResilientLegion

// forward declarations
class InputFile;
class WriteXY;
class ExportGold;
class PennantMapper;

enum MeshFieldID {
    FID_NUMSBAD = 'M' * 100,
    FID_MAPSP1,
    FID_MAPSP1TEMP, // map from sides to points after loading only
    FID_MAPSP2,
    FID_MAPSP2TEMP, // map from sides to points after loading only
    FID_MAPSZ,
    FID_MAPSS3,
    FID_MAPSS4,
    FID_MAPSP1REG,
    FID_MAPSP2REG,
    FID_MAPLOAD2DENSE, // map from load points to dense points
    FID_ZNUMP,
    FID_PX,
    FID_EX,
    FID_ZX,
    FID_PXP,
    FID_EXP,
    FID_ZXP,
    FID_PX0,
    FID_SAREA,
    FID_SVOL,
    FID_ZAREA,
    FID_ZVOL,
    FID_SAREAP,
    FID_SVOLP,
    FID_ZAREAP,
    FID_ZVOLP,
    FID_ZVOL0,
    FID_SSURFP,
    FID_ELEN,
    FID_SMF,
    FID_ZDL,
    FID_PIECE,
    FID_COUNT,
    FID_RANGE
};

enum HydroFieldID {
    FID_DTREC = 'H' * 100,
    FID_PU,
    FID_PU0,
    FID_PMASWT,
    FID_PF,
    FID_PAP,
    FID_ZM,
    FID_ZR,
    FID_ZRP,
    FID_ZE,
    FID_ZETOT,
    FID_ZW,
    FID_ZWRATE,
    FID_ZP,
    FID_ZSS,
    FID_ZDU,
    FID_SFP,
    FID_SFQ,
    FID_SFT
};

enum QCSFieldID {
    FID_CAREA = 'Q' * 100,
    FID_CEVOL,
    FID_CDU,
    FID_CDIV,
    FID_CCOS,
    FID_CQE1,
    FID_CQE2,
    FID_ZUC,
    FID_CRMU,
    FID_CW,
    FID_ZTMP
};

enum MeshTaskID {
    TID_SUMTOPTSDBL = 'M' * 100,
    TID_CALCCTRS,
    TID_CALCVOLS,
    TID_CALCSIDEFRACS,
    TID_CALCSURFVECS,
    TID_CALCEDGELEN,
    TID_CALCCHARLEN,
    TID_COUNTPOINTS,
    TID_CALCRANGES,
    TID_COMPACTPOINTS,
    TID_CALCOWNERS,
    TID_CHECKBADSIDES,
    TID_TEMPGATHER,
    TID_WRITE
};

enum MeshOpID {
    OPID_SUMINT = 'M' * 100,
    OPID_SUMDBL,
    OPID_SUMDBL2,
    OPID_MINDBL,
    OPID_MAXDBL
};

enum ShardingID {
    PENNANT_SHARD_ID = 1,
};

// atomic versions of lhs += rhs
template <typename T> __CUDA_HD__
inline void atomic_add(T& lhs, const T& rhs);

// atomic versions of lhs = min(lhs, rhs)
template <typename T> __CUDA_HD__
inline void atomic_min(T& lhs, const T& rhs);

// atomic versions of lhs = max(lhs, rhs)
template <typename T> __CUDA_HD__
inline void atomic_max(T& lhs, const T& rhs);

template <> __CUDA_HD__
inline void atomic_add(int& lhs, const int& rhs) {
#ifdef __CUDA_ARCH__
    atomicAdd(&lhs, rhs);
#else
    __sync_add_and_fetch(&lhs, rhs);
#endif
}


template <> __CUDA_HD__
inline void atomic_add(double& lhs, const double& rhs) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ < 600
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)&lhs;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(rhs +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
#else
    atomicAdd(&lhs, rhs);
#endif
#else
    long long *target = (long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = oldval.as_float + rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
}


template <> __CUDA_HD__
inline void atomic_add(double2& lhs, const double2& rhs) {
    atomic_add(lhs.x, rhs.x);
    atomic_add(lhs.y, rhs.y);
}


template <> __CUDA_HD__
inline void atomic_min(double& lhs, const double& rhs) {
#ifdef __CUDA_ARCH__
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)&lhs;
#if __CUDA_ARCH__ < 350
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong((__longlong_as_double(assumed) < rhs) ? 
              __longlong_as_double(assumed) : rhs));
    } while (assumed != old);
#else
    atomicMin(address_as_ull, __double_as_longlong(rhs));
#endif
#else
    long long *target = (long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = (oldval.as_float < rhs) ? oldval.as_float : rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
}


template <> __CUDA_HD__
inline void atomic_max(double& lhs, const double& rhs) {
#ifdef __CUDA_ARCH__
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)&lhs;
#if __CUDA_ARCH__ < 350
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong((__longlong_as_double(assumed) > rhs) ? 
              __longlong_as_double(assumed) : rhs));
    } while (assumed != old);
#else
    atomicMax(address_as_ull, __double_as_longlong(rhs));
#endif
#else
    long long *target = (long long *)&lhs;
    union { long long as_int; double as_float; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_float = (oldval.as_float > rhs) ? oldval.as_float : rhs;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
#endif
}

// helper struct for reduction ops
template <typename T, bool EXCLUSIVE>
struct ReduceHelper {
    __CUDA_HD__
    static void addTo(T& lhs, const T& rhs) { lhs += rhs; }
    __CUDA_HD__
    static void minOf(T& lhs, const T& rhs) {
        lhs = (lhs < rhs) ? lhs : rhs;
    }
    __CUDA_HD__
    static void maxOf(T& lhs, const T& rhs) {
        lhs = (lhs > rhs) ? lhs : rhs;
    }
};

// if not exclusive, use an atomic operation
template <typename T>
struct ReduceHelper<T, false> {
    __CUDA_HD__
    static void addTo(T& lhs, const T& rhs) 
    { 
      atomic_add(lhs, rhs); 
    }
    __CUDA_HD__
    static void minOf(T& lhs, const T& rhs) 
    { 
      atomic_min(lhs, rhs); 
    }
    __CUDA_HD__
    static void maxOf(T& lhs, const T& rhs) 
    { 
      atomic_max(lhs, rhs); 
    }
};

template <typename T>
class SumOp {
public:
    typedef T LHS;
    typedef T RHS;
    static const T identity;

    template <bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS& lhs, RHS rhs)
        { ReduceHelper<T, EXCLUSIVE>::addTo(lhs, rhs); }

    template <bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS& rhs1, RHS rhs2)
        { ReduceHelper<T, EXCLUSIVE>::addTo(rhs1, rhs2); }
};


template <typename T>
class MinOp {
public:
    typedef T LHS;
    typedef T RHS;
    static const T identity;

    template <bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS& lhs, RHS rhs)
        { ReduceHelper<T, EXCLUSIVE>::minOf(lhs, rhs); }

    template <bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS& rhs1, RHS rhs2)
        { ReduceHelper<T, EXCLUSIVE>::minOf(rhs1, rhs2); }
};

template <typename T>
class MaxOp {
public:
    typedef T LHS;
    typedef T RHS;
    static const T identity;

    template <bool EXCLUSIVE> __CUDA_HD__
    static void apply(LHS& lhs, RHS rhs)
        { ReduceHelper<T, EXCLUSIVE>::maxOf(lhs, rhs); }

    template <bool EXCLUSIVE> __CUDA_HD__
    static void fold(RHS& rhs1, RHS rhs2)
        { ReduceHelper<T, EXCLUSIVE>::maxOf(rhs1, rhs2); }
};

class PennantShardingFunctor : public Legion::ShardingFunctor {
public:
  PennantShardingFunctor(const Legion::coord_t numpcx, const Legion::coord_t numpcy);
public:
  virtual Legion::ShardID shard(const Legion::DomainPoint &point,
                                const Legion::Domain &full_space,
                                const size_t total_shards);
protected:
  const Legion::coord_t numpcx;
  const Legion::coord_t numpcy;
  Legion::coord_t nsx;
  Legion::coord_t nsy;
  Legion::coord_t pershardx;
  Legion::coord_t pershardy;
  Legion::coord_t shards;
  bool sharded; 
};

class Mesh {
public:
    struct CalcOwnersArgs {
    public:
        CalcOwnersArgs(Legion::IndexPartition priv, 
                       Legion::IndexPartition shared)
          : ip_private(priv), ip_shared(shared) { }
    public:
        Legion::IndexPartition ip_private;
        Legion::IndexPartition ip_shared;
    };
public:

    // children
    GenMesh* gmesh;

    // parameters
    int chunksize;                 // max size for processing chunks
    std::vector<double> subregion; // bounding box for a subregion
                                   // if nonempty, should have 4 entries:
                                   // xmin, xmax, ymin, ymax

    // mesh variables
    // (See documentation for more details on the mesh
    //  data structures...)
    Legion::coord_t nump, nume, numz, nums, numc;
                       // number of points, edges, zones,
                       // sides, corners, resp.
    int numpcs;        // number of pieces in Legion partition
#if 0
    int* mapsp1;       // maps: side -> points 1 and 2
    int* mapsp2;
    int* mapsz;        // map: side -> zone
    int* mapss3;       // map: side -> previous side
    int* mapss4;       // map: side -> next side

    int* znump;        // number of points in zone

    double2* px;       // point coordinates
    double2* ex;       // edge center coordinates
    double2* zx;       // zone center coordinates

    double* sarea;     // side area
    double* svol;      // side volume
    double* zarea;     // zone area
    double* zvol;      // zone volume

    double* smf;       // side mass fraction

    int numsch;                    // number of side chunks
    std::vector<int> schsfirst;    // start/stop index for side chunks
    std::vector<int> schslast;
    std::vector<int> schzfirst;    // start/stop index for zone chunks
    std::vector<int> schzlast;
    int numpch;                    // number of point chunks
    std::vector<int> pchpfirst;    // start/stop index for point chunks
    std::vector<int> pchplast;
    int numzch;                    // number of zone chunks
    std::vector<int> zchzfirst;    // start/stop index for zone chunks
    std::vector<int> zchzlast;
#endif

    std::vector<int> nodecolors;
    colormap nodemcolors;
    Legion::Context ctx;
    Legion::Runtime* runtime;
    Legion::LogicalRegion lrp, lrz, lrs;
    Legion::LogicalRegion lrglb;
    Legion::LogicalPartition lppall, lpz, lps;
    Legion::LogicalPartition lppprv, lppmstr, lppshr;
    Legion::IndexSpace ispc;
    Legion::IndexPartition ippc;
    Legion::Domain dompc;
                                   // domain of legion pieces

    static std::vector<PennantMapper*> local_mappers;

    Mesh(
            const InputFile* inp,
            const int numpcsa,
            Legion::Context ctxa,
            Legion::Runtime* runtimea);
    ~Mesh();

    template<typename T>
    void getField(
            Legion::LogicalRegion& lr,
            const Legion::FieldID fid,
            T* var,
            const int n);

    template<typename T>
    void setField(
            Legion::LogicalRegion& lr,
            const Legion::FieldID fid,
            const T* var,
            const int n);

    void init();
    
    // write mesh statistics
    void writeStats();

    // write mesh
    void write(
            const std::string& probname,
            const Legion::Future &f_cycle,
            const Legion::Future &f_time);

    static void sumToPointsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcCtrsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // OpenMP variant
    static void calcCtrsOMPTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // GPU variant
    static void calcCtrsGPUTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static int calcVolsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // OpenMP variant
    static int calcVolsOMPTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // GPU variant
    static Legion::DeferredReduction<SumOp<int> > calcVolsGPUTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcSideFracsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcSurfVecsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // OpenMP variant
    static void calcSurfVecsOMPTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // GPU variant
    static void calcSurfVecsGPUTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcEdgeLenTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // OpenMP variant
    static void calcEdgeLenOMPTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // GPU variant
    static void calcEdgeLenGPUTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcCharLenTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // OpenMP variant
    static void calcCharLenOMPTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // GPU variant
    static void calcCharLenGPUTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // compute edge, zone centers
    void calcCtrs(
            const double2* px,
            double2* ex,
            double2* zx,
            const int sfirst,
            const int slast);

    // compute side, corner, zone volumes
    int calcVols(
            const double2* px,
            const double2* zx,
            double* sarea,
            double* svol,
            double* zarea,
            double* zvol,
            const int sfirst,
            const int slast);

    // check to see if previous volume computation had any
    // sides with negative volumes
    void checkBadSides(int cycle, Legion::Future f,
                       Legion::Predicate pred);

    // compute side mass fractions
    void calcSideFracs(
            const double* sarea,
            const double* zarea,
            double* smf,
            const int sfirst,
            const int slast);

    void computeRangesParallel(
            const int numpcs,
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion lr_all_range,
            Legion::LogicalRegion lr_private_range,
            Legion::LogicalPartition lp_private_range,
            Legion::LogicalRegion lr_shared_range,
            Legion::LogicalPartition lp_shared_range,
            Legion::IndexPartition ip_private,
            Legion::IndexPartition ip_shared,
            Legion::IndexSpace is_piece);

    void compactPointsParallel(
            const int numpcs,
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion lr_temp_points,
            Legion::LogicalPartition lp_temp_points,
            Legion::LogicalRegion lr_points,
            Legion::LogicalPartition lp_points,
            Legion::IndexSpace is_piece);

    void calcOwnershipParallel(
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion lr_sides,
            Legion::LogicalPartition lp_sides,
            Legion::IndexPartition ip_private,
            Legion::IndexPartition ip_shared,
            Legion::IndexSpace is_piece);

    void calcCtrsParallel(
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion lr_sides,
            Legion::LogicalPartition lp_sides,
            Legion::LogicalRegion lr_zones,
            Legion::LogicalPartition lp_zones,
            Legion::LogicalRegion lr_points,
            Legion::LogicalPartition lp_points_private,
            Legion::LogicalPartition lp_points_shared,
            Legion::IndexSpace is_piece);

    Legion::Future calcVolsParallel(
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion lr_sides,
            Legion::LogicalPartition lp_sides,
            Legion::LogicalRegion lr_zones,
            Legion::LogicalPartition lp_zones,
            Legion::LogicalRegion lr_points,
            Legion::LogicalPartition lp_points_private,
            Legion::LogicalPartition lp_points_shared,
            Legion::IndexSpace is_piece);

    void calcSideFracsParallel(
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion lr_sides,
            Legion::LogicalPartition lp_sides,
            Legion::LogicalRegion lr_zones,
            Legion::LogicalPartition lp_zones,
            Legion::IndexSpace is_piece);

    static void countPointsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcRangesTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void compactPointsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcOwnersTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void checkBadSidesTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void tempGatherTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void writeTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

}; // class Mesh


template<typename T>
void Mesh::getField(
        Legion::LogicalRegion& lr,
        const Legion::FieldID fid,
        T* var,
        const int n) {
    using namespace ResilientLegion;
    RegionRequirement req(lr, LEGION_READ_ONLY, LEGION_EXCLUSIVE, lr);
    req.add_field(fid);
    InlineLauncher inl(req);
    PhysicalRegion pr = runtime->map_region(ctx, inl);
    pr.wait_until_valid();
    FieldAccessor<LEGION_READ_ONLY,T,1,coord_t,
      Realm::AffineAccessor<T,1,coord_t> > acc(pr, fid);
    const IndexSpace& is = lr.get_index_space();
    
    int i = 0;
    for (PointInDomainIterator<1,coord_t> itr(
          runtime->get_index_space_domain(IndexSpaceT<1,coord_t>(is))); 
          itr(); itr++, i++)
      var[i] = acc[*itr];
    runtime->unmap_region(ctx, pr);
}


template<typename T>
void Mesh::setField(
        Legion::LogicalRegion& lr,
        const Legion::FieldID fid,
        const T* var,
        const int n) {
    using namespace ResilientLegion;
    RegionRequirement req(lr, WRITE_DISCARD, LEGION_EXCLUSIVE, lr);
    req.add_field(fid);
    InlineLauncher inl(req);
    PhysicalRegion pr = runtime->map_region(ctx, inl);
    pr.wait_until_valid();
    FieldAccessor<WRITE_DISCARD,T,1,coord_t,
      Realm::AffineAccessor<T,1,coord_t> > acc(pr, fid);
    const IndexSpace& is = lr.get_index_space();

    int i = 0;
    for (PointInDomainIterator<1,coord_t> itr(
          runtime->get_index_space_domain(IndexSpaceT<1,coord_t>(is)));
          itr(); itr++, i++)
      acc[*itr] = var[i];
    runtime->unmap_region(ctx, pr);
}

// Helper method for computing pieces
static inline void calc_pieces_helper(const Legion::coord_t numpcs,
                                      const Legion::coord_t inx,
                                      const Legion::coord_t iny,
                                      Legion::coord_t &numpcx, 
                                      Legion::coord_t &numpcy)
{
  // Hack for experiments to force same partitioning as Regent
  numpcx = 1;
  numpcy = numpcs;
#if 0
  // pick numpcx, numpcy such that pieces are as close to square
  // as possible
  // we would like:  nzx / numpcx == nzy / numpcy,
  // where numpcx * numpcy = numpcs (total number of pieces)
  // this solves to:  numpcx = sqrt(numpcs * nzx / nzy)
  // we compute this, assuming nzx <= nzy (swap if necessary)
  double nx = static_cast<double>(inx);
  double ny = static_cast<double>(iny);
  const bool swapflag = (nx > ny);
  if (swapflag) std::swap(nx, ny);
  double n = sqrt(numpcs * nx / ny);
  // need to constrain n to be an integer with numpcs % n == 0
  // try rounding n both up and down
  Legion::coord_t n1 = floor(n + 1.e-12);
  n1 = std::max(n1, 1LL);
  while (numpcs % n1 != 0) --n1;
  Legion::coord_t n2 = ceil(n - 1.e-12);
  while (numpcs % n2 != 0) ++n2;
  // pick whichever of n1 and n2 gives blocks closest to square,
  // i.e. gives the shortest long side
  double longside1 = std::max(nx / n1, ny / (numpcs/n1));
  double longside2 = std::max(nx / n2, ny / (numpcs/n2));
  numpcx = (longside1 <= longside2 ? n1 : n2);
  numpcy = numpcs / numpcx;
  if (swapflag) std::swap(numpcx, numpcy);
#endif
}

#endif /* MESH_HH_ */
