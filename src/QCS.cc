/*
 * QCS.cc
 *
 *  Created on: Feb 21, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "QCS.hh"

#include <cmath>

#include "legion.h"

#include "MyLegion.hh"
#include "InputFile.hh"
#include "Vec2.hh"
#include "Mesh.hh"
#include "Hydro.hh"

using namespace std;
using namespace Legion;
using namespace LegionRuntime::Accessor;


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_SETCORNERDIV, "setcornerdiv");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setCornerDivTask>(registrar);
    }
    {
      TaskVariantRegistrar registrar(TID_SETQCNFORCE, "setqcnforce");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setQCnForceTask>(registrar);
    }
    {
      TaskVariantRegistrar registrar(TID_SETFORCEQCS, "setforceqcs");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setForceTask>(registrar);
    }
    {
      TaskVariantRegistrar registrar(TID_SETVELDIFF, "setveldiff");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setVelDiffTask>(registrar);
    }
}
}; // namespace


QCS::QCS(const InputFile* inp, Hydro* h) : hydro(h) {
    qgamma = inp->getDouble("qgamma", 5. / 3.);
    q1 = inp->getDouble("q1", 0.);
    q2 = inp->getDouble("q2", 2.);

    // FieldSpace fsz = hydro->mesh->lrz.get_field_space();
    // FieldAllocator faz = hydro->runtime->create_field_allocator(
    //         hydro->ctx, fsz);
    // faz.allocate_field(sizeof(double2), FID_ZUC);
    // faz.allocate_field(sizeof(double), FID_ZTMP);

    // FieldSpace fss = hydro->mesh->lrs.get_field_space();
    // FieldAllocator fas = hydro->runtime->create_field_allocator(
    //         hydro->ctx, fss);
    // fas.allocate_field(sizeof(double), FID_CAREA);
    // fas.allocate_field(sizeof(double), FID_CEVOL);
    // fas.allocate_field(sizeof(double), FID_CDU);
    // fas.allocate_field(sizeof(double), FID_CDIV);
    // fas.allocate_field(sizeof(double), FID_CCOS);
    // fas.allocate_field(sizeof(double2), FID_CQE1);
    // fas.allocate_field(sizeof(double2), FID_CQE2);
    // fas.allocate_field(sizeof(double), FID_CRMU);
    // fas.allocate_field(sizeof(double), FID_CW);
}

QCS::~QCS() {}


// Routine number [2]  in the full algorithm
//     [2.1] Find the corner divergence
//     [2.2] Compute the cos angle for c
//     [2.3] Find the evolution factor cevol(c)
//           and the Delta u(c) = du(c)
void QCS::setCornerDivTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<ptr_t> acc_mapsp1 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP1);
    MyAccessor<ptr_t> acc_mapsp2 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP2);
    MyAccessor<ptr_t> acc_mapss3 =
        get_accessor<ptr_t>(regions[0], FID_MAPSS3);
    MyAccessor<int> acc_mapsp1reg =
        get_accessor<int>(regions[0], FID_MAPSP1REG);
    MyAccessor<int> acc_mapsp2reg =
        get_accessor<int>(regions[0], FID_MAPSP2REG);
    MyAccessor<double2> acc_ex =
        get_accessor<double2>(regions[0], FID_EXP);
    MyAccessor<double> acc_elen =
        get_accessor<double>(regions[0], FID_ELEN);
    MyAccessor<int> acc_znump =
        get_accessor<int>(regions[1], FID_ZNUMP);
    MyAccessor<double2> acc_zx =
        get_accessor<double2>(regions[1], FID_ZXP);
    MyAccessor<double2> acc_pu[2] = {
        get_accessor<double2>(regions[2], FID_PU0),
        get_accessor<double2>(regions[3], FID_PU0)
    };
    MyAccessor<double2> acc_px[2] = {
        get_accessor<double2>(regions[2], FID_PXP),
        get_accessor<double2>(regions[3], FID_PXP)
    };
    MyAccessor<double2> acc_zuc =
        get_accessor<double2>(regions[4], FID_ZUC);
    MyAccessor<double> acc_carea =
        get_accessor<double>(regions[5], FID_CAREA);
    MyAccessor<double> acc_ccos =
        get_accessor<double>(regions[5], FID_CCOS);
    MyAccessor<double> acc_cdiv =
        get_accessor<double>(regions[5], FID_CDIV);
    MyAccessor<double> acc_cevol =
        get_accessor<double>(regions[5], FID_CEVOL);
    MyAccessor<double> acc_cdu =
        get_accessor<double>(regions[5], FID_CDU);

    // [1] Compute a zone-centered velocity
    const IndexSpace& isz = task->regions[1].region.get_index_space();
    for (IndexIterator itrz(runtime,ctx,isz); itrz.has_next(); )
    {
        ptr_t z = itrz.next();
        acc_zuc.write(z, double2(0., 0.));
    }

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrs(runtime,ctx,iss); itrs.has_next(); )
    {
        ptr_t s = itrs.next();
        ptr_t p = acc_mapsp1.read(s);
        int preg = acc_mapsp1reg.read(s);
        ptr_t z = acc_mapsz.read(s);
        double2 pu = acc_pu[preg].read(p);
        double2 zuc = acc_zuc.read(z);
        int n = acc_znump.read(z);
        zuc += pu / n;
        acc_zuc.write(z, zuc);
    }

    // [2] Divergence at the corner
    for (IndexIterator itrc(runtime,ctx,iss); itrc.has_next();)
    {
        ptr_t c = itrc.next();
        ptr_t s2 = c;
        ptr_t s = acc_mapss3.read(s2);
        // Associated zone, point
        ptr_t z = acc_mapsz.read(s);
        ptr_t p = acc_mapsp2.read(s);
        int preg = acc_mapsp2reg.read(s);
        // Neighboring points
        ptr_t p1 = acc_mapsp1.read(s);
        int p1reg = acc_mapsp1reg.read(s);
        ptr_t p2 = acc_mapsp2.read(s2);
        int p2reg = acc_mapsp2reg.read(s2);

        // Velocities and positions
        // 0 = point p
        double2 up0 = acc_pu[preg].read(p);
        double2 xp0 = acc_px[preg].read(p);
        // 1 = edge e2
        double2 up1 = 0.5 * (up0 + acc_pu[p2reg].read(p2));
        double2 xp1 = acc_ex.read(s2);
        // 2 = zone center z
        double2 up2 = acc_zuc.read(z);
        double2 xp2 = acc_zx.read(z);
        // 3 = edge e1
        double2 up3 = 0.5 * (acc_pu[p1reg].read(p1) + up0);
        double2 xp3 = acc_ex.read(s);

        // compute 2d cartesian volume of corner
        double cvolume = 0.5 * cross(xp2 - xp0, xp3 - xp1);
        acc_carea.write(c, cvolume);

        // compute cosine angle
        double2 v1 = xp3 - xp0;
        double2 v2 = xp1 - xp0;
        double de1 = acc_elen.read(s);
        double de2 = acc_elen.read(s2);
        double minelen = min(de1, de2);
        double ccos = ((minelen < 1.e-12) ?
                0. :
                4. * dot(v1, v2) / (de1 * de2));
        acc_ccos.write(c, ccos);

        // compute divergence of corner
        double cdiv = (cross(up2 - up0, xp3 - xp1) -
                cross(up3 - up1, xp2 - xp0)) /
                (2.0 * cvolume);
        acc_cdiv.write(c, cdiv);

        // compute evolution factor
        double2 dxx1 = 0.5 * (xp1 + xp2 - xp0 - xp3);
        double2 dxx2 = 0.5 * (xp2 + xp3 - xp0 - xp1);
        double dx1 = length(dxx1);
        double dx2 = length(dxx2);

        // average corner-centered velocity
        double2 duav = 0.25 * (up0 + up1 + up2 + up3);

        double test1 = abs(dot(dxx1, duav) * dx2);
        double test2 = abs(dot(dxx2, duav) * dx1);
        double num = (test1 > test2 ? dx1 : dx2);
        double den = (test1 > test2 ? dx2 : dx1);
        double r = num / den;
        double evol = sqrt(4.0 * cvolume * r);
        evol = min(evol, 2.0 * minelen);

        // compute delta velocity
        double dv1 = length2(up1 + up2 - up0 - up3);
        double dv2 = length2(up2 + up3 - up0 - up1);
        double du = sqrt(max(dv1, dv2));

        evol = (cdiv < 0.0 ? evol : 0.);
        du   = (cdiv < 0.0 ? du   : 0.);
        acc_cevol.write(c, evol);
        acc_cdu.write(c, du);
    }
}


// Routine number [4]  in the full algorithm CS2DQforce(...)
void QCS::setQCnForceTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double qgamma = args[0];
    const double q1     = args[1];
    const double q2     = args[2];

    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<ptr_t> acc_mapsp1 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP1);
    MyAccessor<ptr_t> acc_mapsp2 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP2);
    MyAccessor<ptr_t> acc_mapss3 =
        get_accessor<ptr_t>(regions[0], FID_MAPSS3);
    MyAccessor<int> acc_mapsp1reg =
        get_accessor<int>(regions[0], FID_MAPSP1REG);
    MyAccessor<int> acc_mapsp2reg =
        get_accessor<int>(regions[0], FID_MAPSP2REG);
    MyAccessor<double> acc_elen =
        get_accessor<double>(regions[0], FID_ELEN);
    MyAccessor<double> acc_cdiv =
        get_accessor<double>(regions[0], FID_CDIV);
    MyAccessor<double> acc_cdu =
        get_accessor<double>(regions[0], FID_CDU);
    MyAccessor<double> acc_cevol =
        get_accessor<double>(regions[0], FID_CEVOL);
    MyAccessor<double> acc_zrp =
        get_accessor<double>(regions[1], FID_ZRP);
    MyAccessor<double> acc_zss =
        get_accessor<double>(regions[1], FID_ZSS);
    MyAccessor<double2> acc_pu[2] = {
        get_accessor<double2>(regions[2], FID_PU0),
        get_accessor<double2>(regions[3], FID_PU0)
    };
    MyAccessor<double> acc_crmu =
        get_accessor<double>(regions[4], FID_CRMU);
    MyAccessor<double2> acc_cqe1 =
        get_accessor<double2>(regions[4], FID_CQE1);
    MyAccessor<double2> acc_cqe2 =
        get_accessor<double2>(regions[4], FID_CQE2);

    const double gammap1 = qgamma + 1.0;

    // [4.1] Compute the crmu (real Kurapatenko viscous scalar)
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrc(runtime,ctx,iss); itrc.has_next(); )
    {
        ptr_t c = itrc.next();
        ptr_t z = acc_mapsz.read(c);

        // Kurapatenko form of the viscosity
        double cdu = acc_cdu.read(c);
        double ztmp2 = q2 * 0.25 * gammap1 * cdu;
        double zss = acc_zss.read(z);
        double ztmp1 = q1 * zss;
        double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
        // Compute crmu for each corner
        double zrp = acc_zrp.read(z);
        double cevol = acc_cevol.read(c);
        double crmu = zkur * zrp * cevol;
        double cdiv = acc_cdiv.read(c);
        crmu = ((cdiv > 0.0) ? 0. : crmu);
        acc_crmu.write(c, crmu);
    }

    // [4.2] Compute the cqe for each corner
    for (IndexIterator itrc(runtime,ctx,iss); itrc.has_next(); )
    {
        ptr_t c = itrc.next();
        ptr_t s2 = c;
        ptr_t s = acc_mapss3.read(s2);
        ptr_t p = acc_mapsp2.read(s);
        int preg = acc_mapsp2reg.read(s);
        // Associated point 1
        ptr_t p1 = acc_mapsp1.read(s);
        int p1reg = acc_mapsp1reg.read(s);
        // Associated point 2
        ptr_t p2 = acc_mapsp2.read(s2);
        int p2reg = acc_mapsp2reg.read(s2);

        // Compute: cqe(1,2,3)=edge 1, y component (2nd), 3rd corner
        //          cqe(2,1,3)=edge 2, x component (1st)
        double crmu = acc_crmu.read(c);
        double2 pu = acc_pu[preg].read(p);
        double2 pu1 = acc_pu[p1reg].read(p1);
        double elen = acc_elen.read(s);
        double2 cqe1 = crmu * (pu - pu1) / elen;
        acc_cqe1.write(c, cqe1);
        double2 pu2 = acc_pu[p2reg].read(p2);
        double elen2 = acc_elen.read(s2);
        double2 cqe2 = crmu * (pu2 - pu) / elen2;
        acc_cqe2.write(c, cqe2);
    }
}


// Routine number [5]  in the full algorithm CS2DQforce(...)
void QCS::setForceTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    MyAccessor<ptr_t> acc_mapss4 =
        get_accessor<ptr_t>(regions[0], FID_MAPSS4);
    MyAccessor<double> acc_carea =
        get_accessor<double>(regions[0], FID_CAREA);
    MyAccessor<double2> acc_cqe1 =
        get_accessor<double2>(regions[0], FID_CQE1);
    MyAccessor<double2> acc_cqe2 =
        get_accessor<double2>(regions[0], FID_CQE2);
    MyAccessor<double> acc_elen =
        get_accessor<double>(regions[0], FID_ELEN);
    MyAccessor<double> acc_ccos =
        get_accessor<double>(regions[1], FID_CCOS);
    MyAccessor<double> acc_cw =
        get_accessor<double>(regions[2], FID_CW);
    MyAccessor<double2> acc_sfq =
        get_accessor<double2>(regions[2], FID_SFQ);

    // [5.1] Preparation of extra variables
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrc(runtime,ctx,iss); itrc.has_next(); )
    {
        ptr_t c = itrc.next();
        double ccos = acc_ccos.read(c);
        double csin2 = 1.0 - ccos * ccos;
        double carea = acc_carea.read(c);
        double cw = ((csin2 < 1.e-4) ? 0. : carea / csin2);
        acc_cw.write(c, cw);
        ccos      = ((csin2 < 1.e-4) ? 0. : ccos);
        acc_ccos.write(c, ccos);
    }

    // [5.2] Set-Up the forces on corners
    for (IndexIterator itrs(runtime,ctx,iss); itrs.has_next(); )
    {
        ptr_t s = itrs.next();
        // Associated corners 1 and 2
        ptr_t c1 = s;
        ptr_t c2 = acc_mapss4.read(s);
        // Edge length for c1, c2 contribution to s
        double el = acc_elen.read(s);

        double cw1 = acc_cw.read(c1);
        double cw2 = acc_cw.read(c2);
        double ccos1 = acc_ccos.read(c1);
        double ccos2 = acc_ccos.read(c2);
        double2 cqe11 = acc_cqe1.read(c1);
        double2 cqe12 = acc_cqe1.read(c2);
        double2 cqe21 = acc_cqe2.read(c1);
        double2 cqe22 = acc_cqe2.read(c2);
        double2 sfq = (cw1 * (cqe21 + ccos1 * cqe11) +
                       cw2 * (cqe12 + ccos2 * cqe22)) / el;
        acc_sfq.write(s, sfq);
    }
}


// Routine number [6] in the full algorithm
void QCS::setVelDiffTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double q1 = args[0];
    const double q2 = args[1];

    MyAccessor<ptr_t> acc_mapsz =
        get_accessor<ptr_t>(regions[0], FID_MAPSZ);
    MyAccessor<ptr_t> acc_mapsp1 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP1);
    MyAccessor<ptr_t> acc_mapsp2 =
        get_accessor<ptr_t>(regions[0], FID_MAPSP2);
    MyAccessor<int> acc_mapsp1reg =
        get_accessor<int>(regions[0], FID_MAPSP1REG);
    MyAccessor<int> acc_mapsp2reg =
        get_accessor<int>(regions[0], FID_MAPSP2REG);
    MyAccessor<double> acc_elen =
        get_accessor<double>(regions[0], FID_ELEN);
    MyAccessor<double> acc_zss =
        get_accessor<double>(regions[1], FID_ZSS);
    MyAccessor<double2> acc_pu[2] = {
        get_accessor<double2>(regions[2], FID_PU0),
        get_accessor<double2>(regions[3], FID_PU0)
    };
    MyAccessor<double2> acc_px[2] = {
        get_accessor<double2>(regions[2], FID_PXP),
        get_accessor<double2>(regions[3], FID_PXP)
    };
    MyAccessor<double> acc_ztmp =
        get_accessor<double>(regions[4], FID_ZTMP);
    MyAccessor<double> acc_zdu =
        get_accessor<double>(regions[4], FID_ZDU);

    const IndexSpace& isz = task->regions[4].region.get_index_space();
    for (IndexIterator itrz(runtime,ctx,isz); itrz.has_next(); )
    {
        ptr_t z = itrz.next();
        acc_ztmp.write(z, 0.);
    }

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (IndexIterator itrs(runtime,ctx,iss); itrs.has_next();)
    {
        ptr_t s = itrs.next();
        ptr_t p1 = acc_mapsp1.read(s);
        int p1reg = acc_mapsp1reg.read(s);
        ptr_t p2 = acc_mapsp2.read(s);
        int p2reg = acc_mapsp2reg.read(s);
        ptr_t z  = acc_mapsz.read(s);

        double2 px1 = acc_px[p1reg].read(p1);
        double2 px2 = acc_px[p2reg].read(p2);
        double2 pu1 = acc_pu[p1reg].read(p1);
        double2 pu2 = acc_pu[p2reg].read(p2);
        double2 dx  = px2 - px1;
        double2 du  = pu2 - pu1;
        double lenx = acc_elen.read(s);
        double dux = dot(du, dx);
        dux = (lenx > 0. ? abs(dux) / lenx : 0.);

        double ztmp  = acc_ztmp.read(z);
        ztmp = max(ztmp, dux);
        acc_ztmp.write(z, ztmp);
    }

    for (IndexIterator itrz(runtime,ctx,isz); itrz.has_next();)
    {
        ptr_t z = itrz.next();
        double zss  = acc_zss.read(z);
        double ztmp  = acc_ztmp.read(z);
        double zdu = q1 * zss + 2. * q2 * ztmp;
        acc_zdu.write(z, zdu);
    }
}

