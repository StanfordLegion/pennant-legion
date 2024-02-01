/*
 * HydroBC.hh
 *
 *  Created on: Jan 13, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef HYDROBC_HH_
#define HYDROBC_HH_

#include <vector>

#include "legion.h"

#include "Vec2.hh"

// forward declarations
class Mesh;

enum HydroBCFieldID {
    FID_MAPBP = 'B' * 100,
    FID_MAPBPREG
};

enum HydroBCTaskID {
    TID_APPLYFIXEDBC = 'B' * 100,
    TID_COUNTBCPOINTS,
    TID_COUNTBCRANGES,
    TID_CREATEBCMAPS
};


class HydroBC {
public:
    struct CountBCArgs {
    public:
        CountBCArgs(double b, double e, bool x)
          : bound(b), eps(e), xplane(x) { }
    public:
        double bound, eps;
        bool xplane;
    };
public:

    // associated mesh object
    Mesh* mesh;

    int numb;                      // number of bdy points
    double2 vfix;                  // vector perp. to fixed plane
    int* mapbp;                    // map: bdy point -> point
    std::vector<int> pchbfirst;    // start/stop index for bdy pt chunks
    std::vector<int> pchblast;

    Legion::LogicalRegion lrb;
    Legion::LogicalPartition lpb;

    HydroBC(
            Mesh* msh,
            const double2 v,
            const double bound, 
            const bool xplane);

    ~HydroBC();

    static void applyFixedBCTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);
    
    // OpenMP variant
    static void applyFixedBCOMPTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    // GPU variant
    static void applyFixedBCGPUTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void countBCPointsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static Legion::coord_t countBCRangesTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

    static void createBCMapsTask(
            const Legion::Task *task,
            const std::vector<Legion::PhysicalRegion> &regions,
            Legion::Context ctx,
            Legion::Runtime *runtime);

}; // class HydroBC


#endif /* HYDROBC_HH_ */
