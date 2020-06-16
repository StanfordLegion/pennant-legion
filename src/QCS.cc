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


namespace {  // unnamed
static void __attribute__ ((constructor)) registerTasks() {
    {
      TaskVariantRegistrar registrar(TID_SETCORNERDIV, "CPU setcornerdiv");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      add_colocation_constraint(registrar, 2, 3, FID_PXP, FID_PU0);
      Runtime::preregister_task_variant<QCS::setCornerDivTask>(registrar, "setcornerdiv");
    }
    {
      TaskVariantRegistrar registrar(TID_SETQCNFORCE, "CPU setqcnforce");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      add_colocation_constraint(registrar, 2, 3, FID_PU0);
      Runtime::preregister_task_variant<QCS::setQCnForceTask>(registrar, "setqcnforce");
    }
    {
      TaskVariantRegistrar registrar(TID_SETFORCEQCS, "CPU setforceqcs");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setForceTask>(registrar, "setforceqcs");
    }
    {
      TaskVariantRegistrar registrar(TID_SETVELDIFF, "CPU setveldiff");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      registrar.set_leaf();
      add_colocation_constraint(registrar, 2, 3, FID_PXP, FID_PU0);
      Runtime::preregister_task_variant<QCS::setVelDiffTask>(registrar, "setveldiff");
    }
    {
      TaskVariantRegistrar registrar(TID_SETCORNERDIV, "OMP setcornerdiv");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      add_colocation_constraint(registrar, 2, 3, FID_PXP, FID_PU0);
      Runtime::preregister_task_variant<QCS::setCornerDivOMPTask>(registrar, "setcornerdiv");
    }
    {
      TaskVariantRegistrar registrar(TID_SETQCNFORCE, "OMP setqcnforce");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      add_colocation_constraint(registrar, 2, 3, FID_PU0);
      Runtime::preregister_task_variant<QCS::setQCnForceOMPTask>(registrar, "setqcnforce");
    }
    {
      TaskVariantRegistrar registrar(TID_SETFORCEQCS, "OMP setforceqcs");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<QCS::setForceOMPTask>(registrar, "setforceqcs");
    }
    {
      TaskVariantRegistrar registrar(TID_SETVELDIFF, "OMP setveldiff");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      registrar.set_leaf();
      add_colocation_constraint(registrar, 2, 3, FID_PXP, FID_PU0);
      Runtime::preregister_task_variant<QCS::setVelDiffOMPTask>(registrar, "setveldiff");
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
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<double2> acc_ex(regions[0], FID_EXP);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    const AccessorRO<double2> acc_zx(regions[1], FID_ZXP);
    const AccessorMC<double2> acc_pu(regions.begin()+2, regions.begin()+4, FID_PU0);
    const AccessorMC<double2> acc_px(regions.begin()+2, regions.begin()+4, FID_PXP);
    const AccessorWD<double2> acc_zuc(regions[4], FID_ZUC);
    const AccessorWD<double> acc_carea(regions[5], FID_CAREA);
    const AccessorWD<double> acc_ccos(regions[5], FID_CCOS);
    const AccessorWD<double> acc_cdiv(regions[5], FID_CDIV);
    const AccessorWD<double> acc_cevol(regions[5], FID_CEVOL);
    const AccessorWD<double> acc_cdu(regions[5], FID_CDU);

    // [1] Compute a zone-centered velocity
    const IndexSpace& isz = task->regions[1].region.get_index_space();
    for (PointIterator itz(runtime, isz); itz(); itz++)
        acc_zuc[*itz] = double2(0., 0.);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer s = *its;
        const Pointer p = acc_mapsp1[s];
        const Pointer z = acc_mapsz[s];
        const double2 pu = acc_pu[p];
        const double2 zuc = acc_zuc[z];
        const int n = acc_znump[z];
        acc_zuc[z] = zuc + pu / n;
    }

    // [2] Divergence at the corner
    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer c = *its;
        const Pointer s2 = c;
        const Pointer s = acc_mapss3[s2];
        // Associated zone, point
        const Pointer z = acc_mapsz[s];
        const Pointer p = acc_mapsp2[s];
        // Neighboring points
        const Pointer p1 = acc_mapsp1[s];
        const Pointer p2 = acc_mapsp2[s2];

        // Velocities and positions
        // 0 = point p
        const double2 up0 = acc_pu[p];
        const double2 xp0 = acc_px[p];
        // 1 = edge e2
        const double2 up1 = 0.5 * (up0 + acc_pu[p2]);
        const double2 xp1 = acc_ex[s2];
        // 2 = zone center z
        const double2 up2 = acc_zuc[z];
        const double2 xp2 = acc_zx[z];
        // 3 = edge e1
        const double2 up3 = 0.5 * (acc_pu[p1] + up0);
        const double2 xp3 = acc_ex[s];

        // compute 2d cartesian volume of corner
        double cvolume = 0.5 * cross(xp2 - xp0, xp3 - xp1);
        acc_carea[c] = cvolume;

        // compute cosine angle
        const double2 v1 = xp3 - xp0;
        const double2 v2 = xp1 - xp0;
        const double de1 = acc_elen[s];
        const double de2 = acc_elen[s2];
        const double minelen = min(de1, de2);
        const double ccos = ((minelen < 1.e-12) ?
                0. :
                4. * dot(v1, v2) / (de1 * de2));
        acc_ccos[c] = ccos;

        // compute divergence of corner
        const double cdiv = (cross(up2 - up0, xp3 - xp1) -
                cross(up3 - up1, xp2 - xp0)) /
                (2.0 * cvolume);
        acc_cdiv[c] = cdiv;

        // compute evolution factor
        const double2 dxx1 = 0.5 * (xp1 + xp2 - xp0 - xp3);
        const double2 dxx2 = 0.5 * (xp2 + xp3 - xp0 - xp1);
        const double dx1 = length(dxx1);
        const double dx2 = length(dxx2);

        // average corner-centered velocity
        const double2 duav = 0.25 * (up0 + up1 + up2 + up3);

        const double test1 = abs(dot(dxx1, duav) * dx2);
        const double test2 = abs(dot(dxx2, duav) * dx1);
        const double num = (test1 > test2 ? dx1 : dx2);
        const double den = (test1 > test2 ? dx2 : dx1);
        const double r = num / den;
        double evol = sqrt(4.0 * cvolume * r);
        evol = min(evol, 2.0 * minelen);

        // compute delta velocity
        const double dv1 = length2(up1 + up2 - up0 - up3);
        const double dv2 = length2(up2 + up3 - up0 - up1);
        double du = sqrt(max(dv1, dv2));

        evol = (cdiv < 0.0 ? evol : 0.);
        du   = (cdiv < 0.0 ? du   : 0.);
        acc_cevol[c] = evol;
        acc_cdu[c] = du;
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

    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<double> acc_cdiv(regions[0], FID_CDIV);
    const AccessorRO<double> acc_cdu(regions[0], FID_CDU);
    const AccessorRO<double> acc_cevol(regions[0], FID_CEVOL);
    const AccessorRO<double> acc_zrp(regions[1], FID_ZRP);
    const AccessorRO<double> acc_zss(regions[1], FID_ZSS);
    const AccessorMC<double2> acc_pu(regions.begin()+2, regions.begin()+4, FID_PU0);
    const AccessorWD<double> acc_crmu(regions[4], FID_CRMU);
    const AccessorWD<double2> acc_cqe1(regions[4], FID_CQE1);
    const AccessorWD<double2> acc_cqe2(regions[4], FID_CQE2);

    const double gammap1 = qgamma + 1.0;

    // [4.1] Compute the crmu (real Kurapatenko viscous scalar)
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer c = *its;
        const Pointer z = acc_mapsz[c];

        // Kurapatenko form of the viscosity
        const double cdu = acc_cdu[c];
        const double ztmp2 = q2 * 0.25 * gammap1 * cdu;
        const double zss = acc_zss[z];
        const double ztmp1 = q1 * zss;
        const double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
        // Compute crmu for each corner
        const double zrp = acc_zrp[z];
        const double cevol = acc_cevol[c];
        const double crmu = zkur * zrp * cevol;
        const double cdiv = acc_cdiv[c];
        acc_crmu[c] = ((cdiv > 0.0) ? 0. : crmu);
    }

    // [4.2] Compute the cqe for each corner
    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer c = *its;
        const Pointer s2 = c;
        const Pointer s = acc_mapss3[s2];
        const Pointer p = acc_mapsp2[s];
        // Associated point 1
        const Pointer p1 = acc_mapsp1[s];
        // Associated point 2
        const Pointer p2 = acc_mapsp2[s2];

        // Compute: cqe(1,2,3)=edge 1, y component (2nd), 3rd corner
        //          cqe(2,1,3)=edge 2, x component (1st)
        const double crmu = acc_crmu[c];
        const double2 pu = acc_pu[p];
        const double2 pu1 = acc_pu[p1];
        const double elen = acc_elen[s];
        const double2 cqe1 = crmu * (pu - pu1) / elen;
        acc_cqe1[c] = cqe1;
        const double2 pu2 = acc_pu[p2];
        const double elen2 = acc_elen[s2];
        const double2 cqe2 = crmu * (pu2 - pu) / elen2;
        acc_cqe2[c] = cqe2;
    }
}


// Routine number [5]  in the full algorithm CS2DQforce(...)
void QCS::setForceTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapss4(regions[0], FID_MAPSS4);
    const AccessorRO<double> acc_carea(regions[0], FID_CAREA);
    const AccessorRO<double2> acc_cqe1(regions[0], FID_CQE1);
    const AccessorRO<double2> acc_cqe2(regions[0], FID_CQE2);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRW<double> acc_ccos(regions[1], FID_CCOS);
    const AccessorWD<double> acc_cw(regions[2], FID_CW);
    const AccessorWD<double2> acc_sfq(regions[2], FID_SFQ);

    // [5.1] Preparation of extra variables
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer c = *its;
        const double ccos = acc_ccos[c];
        const double csin2 = 1.0 - ccos * ccos;
        const double carea = acc_carea[c];
        const double cw = ((csin2 < 1.e-4) ? 0. : carea / csin2);
        acc_cw[c] = cw;
        acc_ccos[c] = ((csin2 < 1.e-4) ? 0. : ccos);
    }

    // [5.2] Set-Up the forces on corners
    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer s = *its;
        // Associated corners 1 and 2
        const Pointer c1 = s;
        const Pointer c2 = acc_mapss4[s];
        // Edge length for c1, c2 contribution to s
        const double el = acc_elen[s];

        const double cw1 = acc_cw[c1];
        const double cw2 = acc_cw[c2];
        const double ccos1 = acc_ccos[c1];
        const double ccos2 = acc_ccos[c2];
        const double2 cqe11 = acc_cqe1[c1];
        const double2 cqe12 = acc_cqe1[c2];
        const double2 cqe21 = acc_cqe2[c1];
        const double2 cqe22 = acc_cqe2[c2];
        const double2 sfq = (cw1 * (cqe21 + ccos1 * cqe11) +
                       cw2 * (cqe12 + ccos2 * cqe22)) / el;
        acc_sfq[s] = sfq;
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

    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<double> acc_zss(regions[1], FID_ZSS);
    const AccessorMC<double2> acc_pu(regions.begin()+2, regions.begin()+4, FID_PU0);
    const AccessorMC<double2> acc_px(regions.begin()+2, regions.begin()+4, FID_PXP);
    const AccessorWD<double> acc_ztmp(regions[4], FID_ZTMP);
    const AccessorWD<double> acc_zdu(regions[4], FID_ZDU);

    const IndexSpace& isz = task->regions[4].region.get_index_space();
    for (PointIterator itz(runtime, isz); itz(); itz++)
        acc_ztmp[*itz] = 0.;

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    for (PointIterator its(runtime, iss); its(); its++)
    {
        const Pointer s = *its;
        const Pointer p1 = acc_mapsp1[s];
        const Pointer p2 = acc_mapsp2[s];
        const Pointer z = acc_mapsz[s];

        const double2 px1 = acc_px[p1];
        const double2 px2 = acc_px[p2];
        const double2 pu1 = acc_pu[p1];
        const double2 pu2 = acc_pu[p2];
        const double2 dx  = px2 - px1;
        const double2 du  = pu2 - pu1;
        const double lenx = acc_elen[s];
        double dux = dot(du, dx);
        dux = (lenx > 0. ? abs(dux) / lenx : 0.);

        double ztmp  = acc_ztmp[z];
        ztmp = max(ztmp, dux);
        acc_ztmp[z] = ztmp;
    }

    for (PointIterator itz(runtime, isz); itz(); itz++)
    {
        const Pointer z = *itz;
        const double zss  = acc_zss[z];
        const double ztmp  = acc_ztmp[z];
        const double zdu = q1 * zss + 2. * q2 * ztmp;
        acc_zdu[z] = zdu;
    }
}

// Routine number [2]  in the full algorithm
//     [2.1] Find the corner divergence
//     [2.2] Compute the cos angle for c
//     [2.3] Find the evolution factor cevol(c)
//           and the Delta u(c) = du(c)
void QCS::setCornerDivOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<double2> acc_ex(regions[0], FID_EXP);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<int> acc_znump(regions[1], FID_ZNUMP);
    const AccessorRO<double2> acc_zx(regions[1], FID_ZXP);
    const AccessorMC<double2> acc_pu(regions.begin()+2, regions.begin()+4, FID_PU0);
    const AccessorMC<double2> acc_px(regions.begin()+2, regions.begin()+4, FID_PXP);
    const AccessorWD<double2> acc_zuc(regions[4], FID_ZUC);
    const AccessorWD<double> acc_carea(regions[5], FID_CAREA);
    const AccessorRW<double> acc_ccos(regions[5], FID_CCOS);
    const AccessorWD<double> acc_cdiv(regions[5], FID_CDIV);
    const AccessorWD<double> acc_cevol(regions[5], FID_CEVOL);
    const AccessorWD<double> acc_cdu(regions[5], FID_CDU);

    // [1] Compute a zone-centered velocity
    const IndexSpace& isz = task->regions[1].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
        acc_zuc[z] = double2(0., 0.);

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer p = acc_mapsp1[s];
        const Pointer z = acc_mapsz[s];
        const double2 pu = acc_pu[p];
        const int n = acc_znump[z];
        SumOp<double2>::apply<false/*exclusive*/>(acc_zuc[z], pu / n);
    }

    // [2] Divergence at the corner
    #pragma omp parallel for
    for (coord_t c = rects.lo[0]; c <= rects.hi[0]; c++)
    {
        const Pointer s2 = c;
        const Pointer s = acc_mapss3[s2];
        // Associated zone, point
        const Pointer z = acc_mapsz[s];
        const Pointer p = acc_mapsp2[s];
        // Neighboring points
        const Pointer p1 = acc_mapsp1[s];
        const Pointer p2 = acc_mapsp2[s2];

        // Velocities and positions
        // 0 = point p
        const double2 up0 = acc_pu[p];
        const double2 xp0 = acc_px[p];
        // 1 = edge e2
        const double2 up1 = 0.5 * (up0 + acc_pu[p2]);
        const double2 xp1 = acc_ex[s2];
        // 2 = zone center z
        const double2 up2 = acc_zuc[z];
        const double2 xp2 = acc_zx[z];
        // 3 = edge e1
        const double2 up3 = 0.5 * (acc_pu[p1] + up0);
        const double2 xp3 = acc_ex[s];

        // compute 2d cartesian volume of corner
        double cvolume = 0.5 * cross(xp2 - xp0, xp3 - xp1);
        acc_carea[c] = cvolume;

        // compute cosine angle
        const double2 v1 = xp3 - xp0;
        const double2 v2 = xp1 - xp0;
        const double de1 = acc_elen[s];
        const double de2 = acc_elen[s2];
        const double minelen = min(de1, de2);
        const double ccos = ((minelen < 1.e-12) ?
                0. :
                4. * dot(v1, v2) / (de1 * de2));
        acc_ccos[c] = ccos;

        // compute divergence of corner
        const double cdiv = (cross(up2 - up0, xp3 - xp1) -
                cross(up3 - up1, xp2 - xp0)) /
                (2.0 * cvolume);
        acc_cdiv[c] = cdiv;

        // compute evolution factor
        const double2 dxx1 = 0.5 * (xp1 + xp2 - xp0 - xp3);
        const double2 dxx2 = 0.5 * (xp2 + xp3 - xp0 - xp1);
        const double dx1 = length(dxx1);
        const double dx2 = length(dxx2);

        // average corner-centered velocity
        const double2 duav = 0.25 * (up0 + up1 + up2 + up3);

        const double test1 = abs(dot(dxx1, duav) * dx2);
        const double test2 = abs(dot(dxx2, duav) * dx1);
        const double num = (test1 > test2 ? dx1 : dx2);
        const double den = (test1 > test2 ? dx2 : dx1);
        const double r = num / den;
        double evol = sqrt(4.0 * cvolume * r);
        evol = min(evol, 2.0 * minelen);

        // compute delta velocity
        const double dv1 = length2(up1 + up2 - up0 - up3);
        const double dv2 = length2(up2 + up3 - up0 - up1);
        double du = sqrt(max(dv1, dv2));

        evol = (cdiv < 0.0 ? evol : 0.);
        du   = (cdiv < 0.0 ? du   : 0.);
        acc_cevol[c] = evol;
        acc_cdu[c] = du;
    }
}


// Routine number [4]  in the full algorithm CS2DQforce(...)
void QCS::setQCnForceOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double qgamma = args[0];
    const double q1     = args[1];
    const double q2     = args[2];

    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<Pointer> acc_mapss3(regions[0], FID_MAPSS3);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<double> acc_cdiv(regions[0], FID_CDIV);
    const AccessorRO<double> acc_cdu(regions[0], FID_CDU);
    const AccessorRO<double> acc_cevol(regions[0], FID_CEVOL);
    const AccessorRO<double> acc_zrp(regions[1], FID_ZRP);
    const AccessorRO<double> acc_zss(regions[1], FID_ZSS);
    const AccessorMC<double2> acc_pu(regions.begin()+2, regions.begin()+4, FID_PU0);
    const AccessorWD<double> acc_crmu(regions[4], FID_CRMU);
    const AccessorWD<double2> acc_cqe1(regions[4], FID_CQE1);
    const AccessorWD<double2> acc_cqe2(regions[4], FID_CQE2);

    const double gammap1 = qgamma + 1.0;

    // [4.1] Compute the crmu (real Kurapatenko viscous scalar)
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for
    for (coord_t c = rects.lo[0]; c <= rects.hi[0]; c++)
    {
        const Pointer z = acc_mapsz[c];

        // Kurapatenko form of the viscosity
        const double cdu = acc_cdu[c];
        const double ztmp2 = q2 * 0.25 * gammap1 * cdu;
        const double zss = acc_zss[z];
        const double ztmp1 = q1 * zss;
        const double zkur = ztmp2 + sqrt(ztmp2 * ztmp2 + ztmp1 * ztmp1);
        // Compute crmu for each corner
        const double zrp = acc_zrp[z];
        const double cevol = acc_cevol[c];
        const double crmu = zkur * zrp * cevol;
        const double cdiv = acc_cdiv[c];
        acc_crmu[c] = ((cdiv > 0.0) ? 0. : crmu);
    }

    // [4.2] Compute the cqe for each corner
    #pragma omp parallel for
    for (coord_t c = rects.lo[0]; c <= rects.hi[0]; c++)
    {
        const Pointer s2 = c;
        const Pointer s = acc_mapss3[s2];
        const Pointer p = acc_mapsp2[s];
        // Associated point 1
        const Pointer p1 = acc_mapsp1[s];
        // Associated point 2
        const Pointer p2 = acc_mapsp2[s2];

        // Compute: cqe(1,2,3)=edge 1, y component (2nd), 3rd corner
        //          cqe(2,1,3)=edge 2, x component (1st)
        const double crmu = acc_crmu[c];
        const double2 pu = acc_pu[p];
        const double2 pu1 = acc_pu[p1];
        const double elen = acc_elen[s];
        const double2 cqe1 = crmu * (pu - pu1) / elen;
        acc_cqe1[c] = cqe1;
        const double2 pu2 = acc_pu[p2];
        const double elen2 = acc_elen[s2];
        const double2 cqe2 = crmu * (pu2 - pu) / elen2;
        acc_cqe2[c] = cqe2;
    }
}


// Routine number [5]  in the full algorithm CS2DQforce(...)
void QCS::setForceOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const AccessorRO<Pointer> acc_mapss4(regions[0], FID_MAPSS4);
    const AccessorRO<double> acc_carea(regions[0], FID_CAREA);
    const AccessorRO<double2> acc_cqe1(regions[0], FID_CQE1);
    const AccessorRO<double2> acc_cqe2(regions[0], FID_CQE2);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorWD<double> acc_ccos(regions[1], FID_CCOS);
    const AccessorWD<double> acc_cw(regions[2], FID_CW);
    const AccessorWD<double2> acc_sfq(regions[2], FID_SFQ);

    // [5.1] Preparation of extra variables
    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp paralel for
    for (coord_t c = rects.lo[0]; c <= rects.hi[0]; c++)
    {
        const double ccos = acc_ccos[c];
        const double csin2 = 1.0 - ccos * ccos;
        const double carea = acc_carea[c];
        const double cw = ((csin2 < 1.e-4) ? 0. : carea / csin2);
        acc_cw[c] = cw;
        acc_ccos[c] = ((csin2 < 1.e-4) ? 0. : ccos);
    }

    // [5.2] Set-Up the forces on corners
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        // Associated corners 1 and 2
        const Pointer c1 = s;
        const Pointer c2 = acc_mapss4[s];
        // Edge length for c1, c2 contribution to s
        const double el = acc_elen[s];

        const double cw1 = acc_cw[c1];
        const double cw2 = acc_cw[c2];
        const double ccos1 = acc_ccos[c1];
        const double ccos2 = acc_ccos[c2];
        const double2 cqe11 = acc_cqe1[c1];
        const double2 cqe12 = acc_cqe1[c2];
        const double2 cqe21 = acc_cqe2[c1];
        const double2 cqe22 = acc_cqe2[c2];
        const double2 sfq = (cw1 * (cqe21 + ccos1 * cqe11) +
                       cw2 * (cqe12 + ccos2 * cqe22)) / el;
        acc_sfq[s] = sfq;
    }
}


// Routine number [6] in the full algorithm
void QCS::setVelDiffOMPTask(
        const Task *task,
        const std::vector<PhysicalRegion> &regions,
        Context ctx,
        Runtime *runtime) {
    const double* args = (const double*) task->args;
    const double q1 = args[0];
    const double q2 = args[1];

    const AccessorRO<Pointer> acc_mapsz(regions[0], FID_MAPSZ);
    const AccessorRO<Pointer> acc_mapsp1(regions[0], FID_MAPSP1);
    const AccessorRO<Pointer> acc_mapsp2(regions[0], FID_MAPSP2);
    const AccessorRO<int> acc_mapsp1reg(regions[0], FID_MAPSP1REG);
    const AccessorRO<int> acc_mapsp2reg(regions[0], FID_MAPSP2REG);
    const AccessorRO<double> acc_elen(regions[0], FID_ELEN);
    const AccessorRO<double> acc_zss(regions[1], FID_ZSS);
    const AccessorMC<double2> acc_pu(regions.begin()+2, regions.begin()+4, FID_PU0);
    const AccessorMC<double2> acc_px(regions.begin()+2, regions.begin()+4, FID_PXP);
    const AccessorWD<double> acc_ztmp(regions[4], FID_ZTMP);
    const AccessorWD<double> acc_zdu(regions[4], FID_ZDU);

    const IndexSpace& isz = task->regions[4].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rectz = runtime->get_index_space_domain(isz);
    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
        acc_ztmp[z] = 0.;

    const IndexSpace& iss = task->regions[0].region.get_index_space();
    // This will assert if it is not dense
    const Rect<1> rects = runtime->get_index_space_domain(iss);
    #pragma omp parallel for
    for (coord_t s = rects.lo[0]; s <= rects.hi[0]; s++)
    {
        const Pointer p1 = acc_mapsp1[s];
        const Pointer p2 = acc_mapsp2[s];
        const Pointer z = acc_mapsz[s];

        const double2 px1 = acc_px[p1];
        const double2 px2 = acc_px[p2];
        const double2 pu1 = acc_pu[p1];
        const double2 pu2 = acc_pu[p2];
        const double2 dx  = px2 - px1;
        const double2 du  = pu2 - pu1;
        const double lenx = acc_elen[s];
        double dux = dot(du, dx);
        dux = (lenx > 0. ? abs(dux) / lenx : 0.);

        MaxOp<double>::apply<false/*exclusive*/>(acc_ztmp[z], dux);
    }

    #pragma omp parallel for
    for (coord_t z = rectz.lo[0]; z <= rectz.hi[0]; z++)
    {
        const double zss  = acc_zss[z];
        const double ztmp  = acc_ztmp[z];
        const double zdu = q1 * zss + 2. * q2 * ztmp;
        acc_zdu[z] = zdu;
    }
}

