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
    TID_APPLYFIXEDBC = 'B' * 100
};


class HydroBC {
public:

    // associated mesh object
    Mesh* mesh;

    int numb;                      // number of bdy points
    double2 vfix;                  // vector perp. to fixed plane
    int* mapbp;                    // map: bdy point -> point
    std::vector<int> pchbfirst;    // start/stop index for bdy pt chunks
    std::vector<int> pchblast;

    LegionRuntime::HighLevel::LogicalRegion lrb;
    LegionRuntime::HighLevel::LogicalPartition lpb;

    HydroBC(
            Mesh* msh,
            const double2 v,
            const std::vector<int>& mbp);

    ~HydroBC();

    static void applyFixedBCTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

}; // class HydroBC


#endif /* HYDROBC_HH_ */
