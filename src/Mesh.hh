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

#include <string>
#include <vector>
#include <algorithm>

#include "legion.h"

#include "Vec2.hh"
#include "GenMesh.hh"

// forward declarations
class InputFile;
class WriteXY;
class ExportGold;


enum MeshFieldID {
    FID_NUMSBAD = 'M' * 100,
    FID_MAPSP1,
    FID_MAPSP2,
    FID_MAPSZ,
    FID_MAPSS3,
    FID_MAPSS4,
    FID_MAPSP1REG,
    FID_MAPSP2REG,
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
    FID_ZDL
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
    TID_CALCSURFVECS,
    TID_CALCEDGELEN,
    TID_CALCCHARLEN
};

enum MeshOpID {
    OPID_SUMINT = 'M' * 100,
    OPID_SUMDBL,
    OPID_SUMDBL2,
    OPID_MINDBL
};

// atomic versions of lhs += rhs
template <typename T>
void atomicAdd(T& lhs, const T& rhs);

// atomic versions of lhs = min(lhs, rhs)
template <typename T>
void atomicMin(T& lhs, const T& rhs);

// helper struct for reduction ops
template <typename T, bool EXCLUSIVE>
struct ReduceHelper {
    static void addTo(T& lhs, const T& rhs) { lhs += rhs; }
    static void minOf(T& lhs, const T& rhs) {
        lhs = std::min(lhs, rhs);
    }
};

// if not exclusive, use an atomic operation
template <typename T>
struct ReduceHelper<T, false> {
    static void addTo(T& lhs, const T& rhs) { atomicAdd(lhs, rhs); }
    static void minOf(T& lhs, const T& rhs) { atomicMin(lhs, rhs); }
};

template <typename T>
class SumOp {
public:
    typedef T LHS;
    typedef T RHS;
    static const T identity;

    template <bool EXCLUSIVE>
    static void apply(LHS& lhs, RHS rhs)
        { ReduceHelper<T, EXCLUSIVE>::addTo(lhs, rhs); }

    template <bool EXCLUSIVE>
    static void fold(RHS& rhs1, RHS rhs2)
        { ReduceHelper<T, EXCLUSIVE>::addTo(rhs1, rhs2); }
};


template <typename T>
class MinOp {
public:
    typedef T LHS;
    typedef T RHS;
    static const T identity;

    template <bool EXCLUSIVE>
    static void apply(LHS& lhs, RHS rhs)
        { ReduceHelper<T, EXCLUSIVE>::minOf(lhs, rhs); }

    template <bool EXCLUSIVE>
    static void fold(RHS& rhs1, RHS rhs2)
        { ReduceHelper<T, EXCLUSIVE>::minOf(rhs1, rhs2); }
};


class Mesh {
public:

    // children
    GenMesh* gmesh;
    WriteXY* wxy;
    ExportGold* egold;

    // parameters
    int chunksize;                 // max size for processing chunks
    std::vector<double> subregion; // bounding box for a subregion
                                   // if nonempty, should have 4 entries:
                                   // xmin, xmax, ymin, ymax

    // mesh variables
    // (See documentation for more details on the mesh
    //  data structures...)
    int nump, nume, numz, nums, numc;
                       // number of points, edges, zones,
                       // sides, corners, resp.
    int numpcs;        // number of pieces in Legion partition
    int numsbad;       // number of bad sides (negative volume)
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

    std::vector<int> nodecolors;
    colormap nodemcolors;
    Legion::Context ctx;
    Legion::Runtime* runtime;
    Legion::LogicalRegion lrp, lrz, lrs;
    Legion::LogicalRegion lrglb;
    Legion::LogicalPartition lppall, lpz, lps;
    Legion::LogicalPartition lppprv, lppmstr, lppshr;
    Legion::Domain dompc;
                                   // domain of legion pieces
    Legion::FutureMap fmapcv;
                                   // future map for calcVolsTask

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

    template<typename Op>
    typename Op::LHS reduceFutureMap(
            Legion::FutureMap& fmap);

    void init();

    // populate mapping arrays
    void initSides(
            std::vector<int>& cellstart,
            std::vector<int>& cellsize,
            std::vector<int>& cellnodes);

    // populate chunk information
    void initChunks();

    // write mesh statistics
    void writeStats();

    // write mesh
    void write(
            const std::string& probname,
            const int cycle,
            const double time,
            const double* zr,
            const double* ze,
            const double* zp);

    // find plane with constant x, y value
    std::vector<int> getXPlane(const double c);
    std::vector<int> getYPlane(const double c);

    // compute chunks for a given plane
    void getPlaneChunks(
            const int numb,
            const int* mapbp,
            std::vector<int>& pchbfirst,
            std::vector<int>& pchblast);

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

    static int calcVolsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcSurfVecsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcEdgeLenTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void calcCharLenTask(
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
    void calcVols(
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
    void checkBadSides();

    // compute side mass fractions
    void calcSideFracs(
            const double* sarea,
            const double* zarea,
            double* smf,
            const int sfirst,
            const int slast);

}; // class Mesh


template<typename T>
void Mesh::getField(
        Legion::LogicalRegion& lr,
        const Legion::FieldID fid,
        T* var,
        const int n) {
    using namespace Legion;
    RegionRequirement req(lr, READ_ONLY, EXCLUSIVE, lr);
    req.add_field(fid);
    InlineLauncher inl(req);
    PhysicalRegion pr = runtime->map_region(ctx, inl);
    pr.wait_until_valid();
    FieldAccessor<READ_ONLY,T,1,coord_t,
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
    using namespace Legion;
    RegionRequirement req(lr, WRITE_DISCARD, EXCLUSIVE, lr);
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


template<typename Op>
typename Op::LHS Mesh::reduceFutureMap(
        Legion::FutureMap& fmap) {
    using namespace Legion;
    typedef typename Op::LHS LHS;
    LHS val = Op::identity;
    for (Domain::DomainPointIterator itrpc(dompc); itrpc; itrpc++) {
        Op::template apply<true>(val, fmap.get_result<LHS>(itrpc.p));
    }
    return val;
}


#endif /* MESH_HH_ */
