/*
 * GenMesh.hh
 *
 *  Created on: Jun 4, 2013
 *      Author: cferenba
 *
 * Copyright (c) 2013, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef GENMESH_HH_
#define GENMESH_HH_

#include <string>
#include <vector>
#include <map>

#include "resilience.h"

#include "Vec2.hh"

#define Legion ResilientLegion

// forward declarations
class InputFile;


typedef std::map<int, std::vector<int> > colormap;

const int MULTICOLOR = -1;

enum GenMeshTaskID {
    TID_GENPOINTS_RECT = 'G' * 100,
    TID_GENPOINTS_PIE,
    TID_GENPOINTS_HEX,
    TID_GENZONES_RECT,
    TID_GENZONES_PIE,
    TID_GENZONES_HEX,
    TID_GENSIDES_RECT,
    TID_GENSIDES_PIE,
    TID_GENSIDES_HEX,
};

class GenMesh {
public:
    struct GenPointArgs {
    public:
      GenPointArgs(GenMesh *gmesh)
        : nzx(gmesh->nzx), nzy(gmesh->nzy),
          numpcx(gmesh->numpcx), numpcy(gmesh->numpcy),
          lenx(gmesh->lenx), leny(gmesh->leny) { }
    public:
      const Legion::coord_t nzx, nzy;
      const Legion::coord_t numpcx, numpcy;
      const double lenx, leny;
    };
    struct GenZoneArgs {
    public:
      GenZoneArgs(GenMesh *gmesh)
        : nzx(gmesh->nzx), nzy(gmesh->nzy),
          numpcx(gmesh->numpcx), numpcy(gmesh->numpcy) { }
    public:
      const Legion::coord_t nzx, nzy;
      const Legion::coord_t numpcx, numpcy;
    };
    struct GenSideArgs {
    public:
      GenSideArgs(GenMesh *gmesh)
        : nzx(gmesh->nzx), nzy(gmesh->nzy),
          numpcx(gmesh->numpcx), numpcy(gmesh->numpcy) { }
    public:
      const Legion::coord_t nzx, nzy;
      const Legion::coord_t numpcx, numpcy;
    };
public:

    std::string meshtype;       // generated mesh type
    Legion::coord_t nzx, nzy;               // number of zones, in x and y
                                // directions
    double lenx, leny;          // length of mesh sides, in x and y
                                // directions
    Legion::coord_t numpcx, numpcy;         // number of pieces to generate,
    bool pieces;
                                // in x and y directions
    std::vector<int> zxbounds, zybounds;
                                // boundaries of pieces, in x and y
                                // directions

    GenMesh(const InputFile* inp);
    ~GenMesh();

    void generate(
            const int numpcs,
            std::vector<double2>& pointpos,
            std::vector<int>& pointcolors,
            colormap& pointmcolors,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& zonecolors);

    void generateRect(
            std::vector<double2>& pointpos,
            std::vector<int>& pointcolors,
            colormap& pointmcolors,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& zonecolors);

    void generatePie(
            std::vector<double2>& pointpos,
            std::vector<int>& pointcolors,
            colormap& pointmcolors,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& zonecolors);

    void generateHex(
            std::vector<double2>& pointpos,
            std::vector<int>& pointcolors,
            colormap& pointmcolors,
            std::vector<int>& zonestart,
            std::vector<int>& zonesize,
            std::vector<int>& zonepoints,
            std::vector<int>& zonecolors);

    void calcNumPieces(const int numpc);

    Legion::coord_t calcNumPoints(const int numpc);

    Legion::coord_t calcNumZones(const int numpc);

    Legion::coord_t calcNumSides(const int numpc);

    void generatePointsParallel(
            const int numpcs,
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion points_lr,
            Legion::LogicalPartition points_lp,
            Legion::IndexSpace piece_is);

    void generateZonesParallel(
            const int numpcs,
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion zones_lr,
            Legion::LogicalPartition zones_lp,
            Legion::IndexSpace piece_is);

    void generateSidesParallel(
            const int numpcs,
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion sides_lr,
            Legion::LogicalPartition sides_lp,
            Legion::IndexSpace piece_is);

    static void genPointsRect(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void genPointsPie(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void genPointsHex(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void genZonesRect(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void genZonesPie(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void genZonesHex(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void genSidesRect(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void genSidesPie(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void genSidesHex(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

}; // class GenMesh


#endif /* GENMESH_HH_ */
