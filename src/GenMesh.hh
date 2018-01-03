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

#include "legion.h"

#include "Vec2.hh"

// forward declarations
class InputFile;


typedef std::map<int, std::vector<int> > colormap;

const int MULTICOLOR = -1;

enum GenMeshTaskID {
    TID_GENPOINTS_RECT = 'G' * 100,
    TID_GENPOINTS_PIE,
    TID_GENPOINTS_HEX,
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
      const int nzx, nzy;
      const int numpcx, numpcy;
      const double lenx, leny;
    };
    struct GenSideArgs {
      GenSideArgs(GenMesh *gmesh)
        : nzx(gmesh->nzx), nzy(gmesh->nzy),
          numpcx(gmesh->numpcx), numpcy(gmesh->numpcy) { }
    public:
      const int nzx, nzy;
      const int numpcx, numpcy;
    };
public:

    std::string meshtype;       // generated mesh type
    int nzx, nzy;               // number of zones, in x and y
                                // directions
    double lenx, leny;          // length of mesh sides, in x and y
                                // directions
    int numpcx, numpcy;         // number of pieces to generate,
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

    int calcNumPoints(const int numpc);

    int calcNumZones(const int numpc);

    int calcNumSides(const int numpc);

    void generatePointsParallel(
            const int numpcs,
            Legion::Runtime *runtime,
            Legion::Context ctx,
            Legion::LogicalRegion points_lr,
            Legion::LogicalPartition points_lp,
            Legion::IndexSpace piece_is);

    void generateSideMapsParallel(
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
