/*
 * ExportGold.cc
 *
 *  Created on: Mar 1, 2012
 *      Author: cferenba
 *
 * Copyright (c) 2012, Los Alamos National Security, LLC.
 * All rights reserved.
 * Use of this source code is governed by a BSD-style open-source
 * license; see top-level LICENSE file for full license text.
 */

#include "ExportGold.hh"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

#include "Vec2.hh"
#include "Mesh.hh"

using namespace std;


ExportGold::ExportGold(Mesh* m) : mesh(m) {}

ExportGold::~ExportGold() {}


void ExportGold::write(
        const string& basename,
        const int cycle,
        const double time,
        const double* zr,
        const double* ze,
        const double* zp) {

    writeCaseFile(basename);

    sortZones();
    writeGeoFile(basename, cycle, time);

    writeVarFile(basename, "zr", zr);
    writeVarFile(basename, "ze", ze);
    writeVarFile(basename, "zp", zp);

}


void ExportGold::writeCaseFile(
        const string& basename) {

    // open file
    const string filename = basename + ".case";
    ofstream ofs(filename.c_str());
    if (!ofs.good()) {
        cerr << "Cannot open file " << filename << " for writing"
             << endl;
        exit(1);
    }

    // write case info
    ofs << "#" << endl;
    ofs << "# Created by PENNANT" << endl;
    ofs << "#" << endl;

    ofs << "FORMAT" << endl;
    ofs << "type: ensight gold" << endl;

    ofs << "GEOMETRY" << endl;
    ofs << "model: " << basename << ".geo" << endl;

    ofs << "VARIABLE" << endl;
    ofs << "scalar per element: zr " << basename << ".zr" << endl;
    ofs << "scalar per element: ze " << basename << ".ze" << endl;
    ofs << "scalar per element: zp " << basename << ".zp" << endl;

    ofs.close();

}


void ExportGold::writeGeoFile(
        const string& basename,
        const int cycle,
        const double time) {

    // open file
    ofstream ofs;
    const string filename = basename + ".geo";
    ofs.open(filename.c_str());
    if (!ofs.good()) {
        cerr << "Cannot open file " << filename << " for writing"
             << endl;
        exit(1);
    }

    // write general header
    ofs << scientific;
    ofs << "cycle = " << setw(8) << cycle << endl;
    ofs << setprecision(8);
    ofs << "t = " << setw(15) << time << endl;
    ofs << "node id assign" << endl;
    ofs << "element id given" << endl;

    // write header for the one "part" (entire mesh)
    ofs << "part" << endl;
    ofs << setw(10) << 1 << endl;
    ofs << "universe" << endl;

    const int nump = mesh->nump;
    const double2* px = mesh->px;

    // write node info
    ofs << "coordinates" << endl;
    ofs << setw(10) << nump << endl;
    ofs << setprecision(5);
    for (int p = 0; p < nump; ++p)
        ofs << setw(12) << px[p].x << endl;
    for (int p = 0; p < nump; ++p)
        ofs << setw(12) << px[p].y << endl;
    // Ensight expects z-coordinates, so write 0 for those
    for (int p = 0; p < nump; ++p)
        ofs << setw(12) << 0. << endl;

    const int* znump = mesh->znump;
    const int* mapsp1 = mesh->mapsp1;

    const int ntris = tris.size();
    const int nquads = quads.size();
    const int nothers = others.size();

    // gather triangle info
    vector<int> trip(3 * ntris);
    for (int t = 0; t < ntris; ++t) {
        int z = tris[t];
        int sbase = mapzs[z];
        for (int i = 0; i < 3; ++i) {
            trip[t * 3 + i] = mapsp1[sbase + i];
        }
    }

    // write triangles
    if (ntris > 0) {
        ofs << "tria3" << endl;
        ofs << setw(10) << ntris << endl;
        for (int t = 0; t < ntris; ++t)
            ofs << setw(10) << tris[t] + 1 << endl;
        for (int t = 0; t < ntris; ++t) {
            for (int i = 0; i < 3; ++i)
                ofs << setw(10) << trip[t * 3 + i] + 1;
            ofs << endl;
        }
    } // if ntris > 0

    // gather quad info
    vector<int> quadp(4 * nquads);
    for (int q = 0; q < nquads; ++q) {
        int z = quads[q];
        int sbase = mapzs[z];
        for (int i = 0; i < 4; ++i) {
            quadp[q * 4 + i] = mapsp1[sbase + i];
        }
    }

    // write quads
    if (nquads > 0) {
        ofs << "quad4" << endl;
        ofs << setw(10) << nquads << endl;
        for (int q = 0; q < nquads; ++q)
            ofs << setw(10) << quads[q] + 1 << endl;
        for (int q = 0; q < nquads; ++q) {
            for (int i = 0; i < 4; ++i)
                ofs << setw(10) << quadp[q * 4 + i] + 1;
            ofs << endl;
        }
    } // if nquads > 0

    // gather other info
    vector<int> othernump(nothers), otherp;
    for (int n = 0; n < nothers; ++n) {
        int z = others[n];
        int sbase = mapzs[z];
        othernump[n] = znump[z];
        for (int i = 0; i < znump[z]; ++i) {
            otherp.push_back(mapsp1[sbase + i]);
        }
    }

    // write others
    if (nothers > 0) {
        ofs << "nsided" << endl;
        ofs << setw(10) << nothers << endl;
        for (int n = 0; n < nothers; ++n)
            ofs << setw(10) << others[n] + 1 << endl;
        for (int n = 0; n < nothers; ++n)
            ofs << setw(10) << othernump[n] << endl;
        int p = 0;
        for (int n = 0; n < nothers; ++n) {
            for (int i = 0; i < othernump[n]; ++i)
                ofs << setw(10) << otherp[p + i] + 1;
            ofs << endl;
            p += othernump[n];
        }
    } // if nothers > 0

    ofs.close();

}


void ExportGold::writeVarFile(
        const string& basename,
        const string& varname,
        const double* var) {

    // open file
    ofstream ofs;
    const string filename = basename + "." + varname;
    ofs.open(filename.c_str());
    if (!ofs.good()) {
        cerr << "Cannot open file " << filename << " for writing"
             << endl;
        exit(1);
    }

    // write header
    ofs << scientific << setprecision(5);
    ofs << varname << endl;
    ofs << "part" << endl;
    ofs << setw(10) << 1 << endl;

    int ntris = tris.size();
    int nquads = quads.size();
    int nothers = others.size();

    // gather values on triangles
    vector<double> tvar(ntris);
    for (int t = 0; t < ntris; ++t) {
        tvar[t] = var[tris[t]];
    }

    // write values on triangles
    if (ntris > 0) {
        ofs << "tria3" << endl;
        for (int t = 0; t < ntris; ++t) {
            ofs << setw(12) << tvar[t] << endl;
        }
    } // if ntris > 0

    // gather values on quads
    vector<double> qvar(nquads);
    for (int q = 0; q < nquads; ++q) {
        qvar[q] = var[quads[q]];
    }

    // write values on quads
    if (nquads > 0) {
        ofs << "quad4" << endl;
        for (int q = 0; q < nquads; ++q) {
            ofs << setw(12) << qvar[q] << endl;
        }
    } // if nquads > 0

    // gather values on others
    vector<double> ovar(nothers);
    for (int n = 0; n < nothers; ++n) {
        ovar[n] = var[others[n]];
    }

    // write values on others
    if (nothers > 0) {
        ofs << "nsided" << endl;
        for (int n = 0; n < nothers; ++n) {
            ofs << setw(12) << ovar[n] << endl;
        }
    } // if nothers > 0

    ofs.close();

}


void ExportGold::sortZones() {

    const int numz = mesh->numz;
    const int* znump = mesh->znump;

    mapzs.resize(numz);

    // sort zones by size, create an inverse map
    int scount = 0;
    for (int z = 0; z < numz; ++z) {
        int zsize = znump[z];
        if (zsize == 3)
            tris.push_back(z);
        else if (zsize == 4)
            quads.push_back(z);
        else // zsize > 4
            others.push_back(z);
        mapzs[z] = scount;
        scount += zsize;
    } // for z

}

