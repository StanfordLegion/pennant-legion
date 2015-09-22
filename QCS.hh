/*
 * QCS.hh
 *
 *  Created on: Feb 21, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#ifndef QCS_HH_
#define QCS_HH_

#include "legion_types.h"

// forward declarations
class InputFile;
class Hydro;


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

enum QCSTaskID {
    TID_SETCORNERDIV = 'Q' * 100,
    TID_SETQCNFORCE,
    TID_SETFORCEQCS,
    TID_SETVELDIFF
};


class QCS {
public:

    // parent hydro object
    Hydro* hydro;

    double qgamma;                 // gamma coefficient for Q model
    double q1, q2;                 // linear and quadratic coefficients
                                   // for Q model

    QCS(const InputFile* inp, Hydro* h);
    ~QCS();

    static void setCornerDivTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void setQCnForceTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void setForceTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

    static void setVelDiffTask(
            const LegionRuntime::HighLevel::Task *task,
            const std::vector<LegionRuntime::HighLevel::PhysicalRegion> &regions,
            LegionRuntime::HighLevel::Context ctx,
            LegionRuntime::HighLevel::HighLevelRuntime *runtime);

};  // class QCS


#endif /* QCS_HH_ */
