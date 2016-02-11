#include "legion_io.h"
#include "hdf5.h"
#include <time.h>
#include <sys/time.h>
#include <stdio.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

void current_utc_time(struct timespec *ts) {

#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
  #else
  clock_gettime(CLOCK_REALTIME, ts);
  #endif

}

struct task_args_t{
  bool copy_write;
#ifdef LEGIONIO_SERIALIZE
  size_t field_map_size; 
  char field_map_serial[4096];
#endif
};



void copy_values_task(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, HighLevelRuntime *runtime);

#ifdef LEGIONIO_PHASER_TIMERS
void timer_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime);
#endif

void split_path_file(char** p, char** f, const char *pf);


void PersistentRegion_init() {
  HighLevelRuntime::register_legion_task<copy_values_task>(COPY_VALUES_TASK_ID,
                                                           Processor::LOC_PROC,
                                                           true /*single*/, true /*index*/);
  
#ifdef LEGIONIO_PHASER_TIMERS
  HighLevelRuntime::register_legion_task<timer_task>(TIMER_TASK_ID,
                                                     Processor::LOC_PROC,
                                                     true /*single*/, false /*index*/);
#endif
    
}
        
PersistentRegion::PersistentRegion(HighLevelRuntime * runtime) {
    this->runtime = runtime; 
}

#ifdef LEGIONIO_PHASER_TIMERS
void timer_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, HighLevelRuntime *runtime)
{
  struct timespec ts;
  current_utc_time(&ts);   
  std::cout << ((const char*) task->args) << " seconds: " << ts.tv_sec
            << " nanos: " << ts.tv_nsec << std::endl; 
}
#endif

void copy_values_task(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, HighLevelRuntime *runtime)
{
#ifdef LEGIONIO_VERBOSE
  std::cout << "copy_values_task: index_point dim is "
            <<  task->index_point.dim << " [" <<  task->index_point[0]
            << "," << task->index_point[1] << ","
            << task->index_point[2] << "]" << std::endl;
#endif

  Piece piece = * ((Piece*) task->local_args);
  struct task_args_t task_args = *(struct task_args_t*) task->args;
  
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  assert(piece.child_lr == regions[0].get_logical_region());
  
 
#ifdef LEGIONIO_TIMERS 
  struct timespec ts;
  current_utc_time(&ts);   
  char hostname[128];
  gethostname(hostname, sizeof hostname);
  if(task_args.copy_write) {
    std::cout << hostname << " domain point: " << piece.dp
              << "; write begins at:  seconds: " << ts.tv_sec
              << " nanos: " << ts.tv_nsec << std::endl; 
  } else {
    std::cout << "domain point: " << piece.dp
              << "; write ends & read begins at:  seconds: " << ts.tv_sec
              << " nanos: " << ts.tv_nsec << std::endl; 
  }
#endif
  
  std::map<FieldID, std::string> field_string_map;
#ifdef LEGIONIO_SERIALIZE
  Realm::Serialization::FixedBufferDeserializer fdb(task_args.field_map_serial,
                                                    task_args.field_map_size);

  bool ok  = fdb >> field_string_map;
  if(!ok) {
    std::cout << "ERROR in copy_values_task, can't deserialize " << std::endl;
  }
#else 
  field_string_map.insert(std::make_pair(FID_PX, "bam/px"));
  field_string_map.insert(std::make_pair(FID_PXP, "bam/pxp"));
  field_string_map.insert(std::make_pair(FID_PX0, "bam/px0"));
  field_string_map.insert(std::make_pair(FID_PU, "bam/pu"));
  field_string_map.insert(std::make_pair(FID_PU0, "bam/pu0"));
  field_string_map.insert(std::make_pair(FID_PMASWT, "bam/pmaswt"));
  field_string_map.insert(std::make_pair(FID_PF, "bam/pf"));
  field_string_map.insert(std::make_pair(FID_PAP, "bam/pap"));

#endif

#ifdef LEGIONIO_VERBOSE
  std::cout << "field_map_size is : " <<task_args.field_map_size << std::endl;
  std::cout << "field_string_map.size is : " << field_string_map.size() << std::endl;
#endif

  std::map<FieldID, const char*> field_map;
  for (std::map<FieldID, std::string>::const_iterator it = field_string_map.begin();
       it != field_string_map.end(); it++)
  {
#ifdef LEGIONIO_VERBOSE
    std::cout << "inserting field from args into local field_map: " <<
      it->first << " : " << it->second << std::endl;
#endif
    field_map.insert(std::make_pair(it->first, it->second.c_str()));
  }
  

#ifdef LEGIONIO_VERBOSE
  std::cout << "In write_values_task and found my piece!" << std::endl;
  std::cout<< "In write_value_task "  << std::endl;
  
#endif

  runtime->unmap_region(ctx, regions[0]);
      
  PhysicalRegion pr = runtime->attach_hdf5(ctx, piece.shard_name,
      piece.child_lr, piece.child_lr, field_map,
      task_args.copy_write ? LEGION_FILE_READ_WRITE: LEGION_FILE_READ_ONLY);

  runtime->remap_region(ctx, pr);
  pr.wait_until_valid();
  
#ifdef LEGIONIO_VERBOSE 
  std::cout << "after remap and pr is valid " << std::endl;
#endif

  CopyLauncher copy_launcher;

  if(task_args.copy_write) { 
    copy_launcher.add_copy_requirements(
      RegionRequirement(regions[1].get_logical_region(),
                        READ_ONLY, EXCLUSIVE,
                        regions[1].get_logical_region()),
      RegionRequirement(piece.child_lr, WRITE_DISCARD,
                        EXCLUSIVE, piece.child_lr));
  } else {
    copy_launcher.add_copy_requirements(
      RegionRequirement(piece.child_lr, READ_ONLY,
                        EXCLUSIVE, piece.child_lr),
      RegionRequirement(regions[1].get_logical_region(),
                        WRITE_DISCARD, EXCLUSIVE,
                        regions[1].get_logical_region()));
  } 
  
  copy_launcher.add_src_field(0, FID_TEMP);
  copy_launcher.add_dst_field(0, FID_TEMP);
  runtime->issue_copy_operation(ctx, copy_launcher);
  
  runtime->detach_hdf5(ctx, pr);
}


void PersistentRegion::write_persistent_subregions(Context ctx, LogicalRegion src_lr, LogicalPartition src_lp){
  
  ArgumentMap arg_map;
  struct task_args_t task_args;
  task_args.copy_write = true;
#ifdef LEGIONIO_SERIALIZE
  task_args.field_map_size = this->field_map_size;
  memcpy(task_args.field_map_serial, this->field_map_serial, this->field_map_size);
#endif

#ifdef LEGIONIO_VERBOSE
  std::cout << "write_persistent_subregions called with fieldmap_size of " <<
    task_args.field_map_size << std::endl;
  std::cout << "write_persistent_subregions dom is [" << dom.get_rect<1>().lo.x[0]
            << "," << dom.get_rect<1>().lo.x[1] << "]" << std::endl; 
#endif 

#ifdef LEGIONIO_PHASER_TIMERS
  char  start_message[]  = "phaser! write_persistent_subregions: start"; 
  TaskLauncher timer_launcher_start(TIMER_TASK_ID, TaskArgument(start_message, 40));
  char end_message[] = "phaser! write_persistent_subregions: end"; 
  TaskLauncher timer_launcher_end(TIMER_TASK_ID, TaskArgument(end_message, 40));
#endif
  
  IndexLauncher write_launcher(COPY_VALUES_TASK_ID, this->dom,
#ifdef LEGIONIO_SERIALIZE
  TaskArgument(&task_args,
               sizeof(task_args)+task_args.field_map_size),
#else
  TaskArgument(&task_args, sizeof(task_args)), 
#endif
  arg_map);

  for(std::vector<Piece>::iterator itr = this->pieces.begin(); 
      itr != this->pieces.end(); itr++) {
    Piece piece = *itr;
    std::cout << "Inserting piece as argument to domain point: [" <<
      piece.dp[0] << "," << piece.dp[1] << "]" << std::endl; 
    arg_map.set_point(piece.dp, TaskArgument(&piece, sizeof(Piece)));
  }
  
#ifdef LEGIONIO_PHASER_TIMERS
  timer_launcher_start.add_arrival_barrier(this->pb_timer);
  runtime->execute_task(ctx, timer_launcher_start);
  write_launcher.add_wait_barrier(this->pb_timer);  
  write_launcher.add_arrival_barrier(this->pb_write);
#endif
  
  write_launcher.add_region_requirement(
    RegionRequirement(this->lp,
		      0 /*no projection */, 
		      READ_WRITE, EXCLUSIVE, this->parent_lr));
  
  write_launcher.add_region_requirement(
    RegionRequirement(src_lp,
		      0 /* No prjections */, 
		      READ_WRITE, EXCLUSIVE, src_lr));
  
  /* setup region requirements using field map */ 
  for(std::map<FieldID, std::string>::iterator iterator = this->field_map.begin(); iterator != this->field_map.end(); iterator++) {
    FieldID fid = iterator->first;
    std::cout << "write_persistent_subregions: adding field : " << fid << std::endl;
    write_launcher.region_requirements[0].add_field(fid, false /* no instance required */);
    write_launcher.region_requirements[1].add_field(fid);
  }
  runtime->execute_index_space(ctx, write_launcher);
  
#ifdef LEGIONIO_PHASER_TIMERS
  this->pb_write = runtime->advance_phase_barrier(ctx, this->pb_write);
  timer_launcher_end.add_wait_barrier(this->pb_write);
  runtime->execute_task(ctx, timer_launcher_end);
#endif
}

						   

void PersistentRegion::read_persistent_subregions(Context ctx, LogicalRegion src_lr, LogicalPartition src_lp){
  
  ArgumentMap arg_map;

  struct task_args_t task_args;
  task_args.copy_write = false;
#ifdef LEGIONIO_SERIALIZE
  task_args.field_map_size = this->field_map_size;
//  std::cout << "write_persistent_subregions: setting field_map_size to: " << this->field_map_size << std::endl;
  memcpy(task_args.field_map_serial, this->field_map_serial, this->field_map_size);
  //std::cout << "task_args size is: " << sizeof(task_args) << std::endl;
#endif
  IndexLauncher read_launcher(COPY_VALUES_TASK_ID, this->dom,
#ifdef LEGIONIO_SERIALIZE
    TaskArgument(&task_args, sizeof(task_args)+task_args.field_map_size),
#else
    TaskArgument(&task_args, sizeof(task_args)),
#endif
                              arg_map);
  
  for(std::vector<Piece>::iterator itr = this->pieces.begin(); 
      itr != this->pieces.end(); itr++) {
    Piece piece = *itr;
    arg_map.set_point(piece.dp, TaskArgument(&piece, sizeof(Piece)));
  }
  
  
  read_launcher.add_region_requirement(
    RegionRequirement(this->lp,
		      0 /*no projection */, 
		      READ_WRITE, EXCLUSIVE, this->parent_lr));
  
  read_launcher.add_region_requirement(
    RegionRequirement(src_lp,
		      0 /* No prjections */, 
		      READ_WRITE, EXCLUSIVE, src_lr));
  
  /* setup region requirements using field map */ 
  for(std::map<FieldID, std::string>::iterator iterator = this->field_map.begin(); iterator != this->field_map.end(); iterator++) {
    FieldID fid = iterator->first;
    read_launcher.region_requirements[0].add_field(fid, false /* no instance required */);
    read_launcher.region_requirements[1].add_field(fid);
  }
  runtime->execute_index_space(ctx, read_launcher); 
}


// This version is used for structured index spaces 
void PersistentRegion::create_persistent_subregions(Context ctx,
    const char *name, LogicalRegion parent_lr, LogicalPartition lp,
    Domain dom, std::map<FieldID, std::string> &field_map)
{
  hid_t link_file_id, shard_group_id, shard_ds_id, dataspace_id, dtype_id, shard_file_id, attr_ds_id, link_group_id, link_group_2_id;
  herr_t status;
  link_file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  dtype_id  = H5Tcopy (H5T_NATIVE_DOUBLE);

  this->lp = lp;
  this->parent_lr = parent_lr;
  this->field_map = field_map;
  this->dom = dom; 
  
#ifdef LEGIONIO_SERIALIZE
  std::map<FieldID, std::string> field_map_des;
  Realm::Serialization::DynamicBufferSerializer dbs(0);
  dbs << field_map;
  this->field_map_size = dbs.bytes_used();
  this->field_map_serial = dbs.detach_buffer();
#endif
  //  Realm::Serialization::FixedBufferDeserializer fbd(this->field_map_serial, this->field_map_size);
  //  bool ok = fbd >> field_map_des;
  //  if(ok) { 
  //    std::cout << " In create_persistent_subregions and I deserialized!: " <<
  //      ok << std::endl;
  //  }

  int i = 0;
  for (Domain::DomainPointIterator itr(dom); itr; itr++) {
    pieces.push_back(Piece());
    DomainPoint dp = itr.p;
    pieces[i].dp = dp;
    pieces[i].child_lr = runtime->get_logical_subregion_by_color(ctx, lp, dp);
    pieces[i].parent_lr = parent_lr;
    IndexSpace is = runtime->get_index_subspace(ctx, lp.get_index_partition(), dp); 
    FieldSpace fs = pieces[i].child_lr.get_field_space();
    Domain d = runtime->get_index_space_domain(ctx, is);
    int dim = d.get_dim();

#ifdef LEGIONIO_VERBOSE
    std::cout << "Found a logical region:  Dimension " << dim <<  std::endl;
#endif
    int x_min = 0, y_min = 0, z_min = 0,
        x_max = 0, y_max = 0, z_max = 0;

    int *shard_dims;
    std::ostringstream ds_name_stream;


    switch(dim) {
      case 2:
        {
          ds_name_stream <<  pieces[i].dp[0] ;
          sprintf(pieces[i].shard_name, "%d-%d-%s",
              pieces[i].dp[0], pieces[i].dp[1], name); 

          x_min = d.get_rect<2>().lo.x[0];
          y_min = d.get_rect<2>().lo.x[1];
          x_max = d.get_rect<2>().hi.x[0];
          y_max = d.get_rect<2>().hi.x[1];

#ifdef LEGIONIO_VERBOSE 
          std::cout << "domain rect is: [[" << x_min << "," << y_min
            << "],[" << x_max  << "," << y_max << "]]" << std::endl; 
#endif

          hsize_t dims[2];
          dims[0] = x_max-x_min+1;
          dims[1] = y_max-y_min+1;
          dataspace_id = H5Screate_simple(2, dims, NULL);

          dims[0] = 2;
          dims[1] = 2; 
          attr_ds_id = H5Screate_simple(2, dims, NULL);

          shard_dims = (int*) malloc(4*sizeof(int)); 
          shard_dims[0] = x_min;
          shard_dims[1] = y_min;
          shard_dims[2] = x_max;
          shard_dims[3] = y_max;
        }
        break;

      case 3:
        {
          ds_name_stream <<  pieces[i].dp[0] << "-" <<
            pieces[i].dp[1] << "-" << pieces[i].dp[2];
          sprintf(pieces[i].shard_name, "%d-%d-%d-%s",
              pieces[i].dp[0], pieces[i].dp[1],
              pieces[i].dp[2], name);

          x_min = d.get_rect<3>().lo.x[0];
          y_min = d.get_rect<3>().lo.x[1];
          z_min = d.get_rect<3>().lo.x[2];
          x_max = d.get_rect<3>().hi.x[0];
          y_max = d.get_rect<3>().hi.x[1];
          z_max = d.get_rect<3>().hi.x[2];

          hsize_t dims[3];
          dims[0] = x_max-x_min+1;
          dims[1] = y_max-y_min+1;
          dims[2] = z_max-z_min+1;
          dataspace_id = H5Screate_simple(3, dims, NULL);

          dims[0] = 2;
          dims[1] = 2; 
          dims[2] = 2; 
          attr_ds_id = H5Screate_simple(3, dims, NULL);

          shard_dims = (int*) malloc(6*sizeof(int)); 
          shard_dims[0] = x_min;
          shard_dims[1] = y_min;
          shard_dims[2] = z_min;
          shard_dims[3] = x_max;
          shard_dims[4] = y_max;
          shard_dims[5] = z_max;
        }
        break;

      default:
        assert(false);
    }

    shard_file_id = H5Fcreate(pieces[i].shard_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    for (std::map<FieldID, std::string>::const_iterator iterator = field_map.begin();
         iterator != field_map.end(); iterator++) {
      FieldID fid = iterator->first;

      char* ds;
      char* gp;
      split_path_file(&gp, &ds, iterator->second.c_str());

      size_t field_size = runtime->get_field_size(ctx, fs, fid);
      status = H5Tset_size(dtype_id, field_size);

      if(H5Lexists(shard_file_id, gp, H5P_DEFAULT)) { 
        shard_group_id = H5Gopen2(shard_file_id, gp, H5P_DEFAULT);
      } else { 
        shard_group_id = H5Gcreate2(shard_file_id, gp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      } 

      if(H5Lexists(shard_group_id, ds, H5P_DEFAULT)) { 
        shard_ds_id = H5Dopen2(shard_group_id, ds, H5P_DEFAULT); 
      } else { 
        shard_ds_id = H5Dcreate2(shard_group_id, ds, H5T_NATIVE_DOUBLE,
            dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
            H5P_DEFAULT);
      }

      if(H5Lexists(link_file_id, gp, H5P_DEFAULT) && H5Lexists(link_file_id, iterator->second.c_str(), H5P_DEFAULT)) {
        link_group_2_id = H5Gopen2(link_file_id, iterator->second.c_str(), H5P_DEFAULT);
      } else {
        link_group_id = H5Gcreate2(link_file_id, gp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        link_group_2_id = H5Gcreate2(link_group_id, ds, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Gclose(link_group_id);
      }

      hid_t attr_id = H5Acreate2(shard_ds_id, "dims", H5T_NATIVE_INT, attr_ds_id,
          H5P_DEFAULT, H5P_DEFAULT);

      status = H5Awrite(attr_id, H5T_NATIVE_INT, shard_dims);
      H5Aclose(attr_id);

      H5Dclose(shard_ds_id);
      H5Gclose(shard_group_id);
      H5Fclose(shard_file_id);

      status = H5Lcreate_external(pieces[i].shard_name, iterator->second.c_str(),
          link_group_2_id, ds_name_stream.str().c_str(),
          H5P_DEFAULT, H5P_DEFAULT);

      shard_file_id = H5Fopen(pieces[i].shard_name, H5F_ACC_RDWR, H5P_DEFAULT);      
      H5Gclose(link_group_2_id);
    }
    H5Fclose(shard_file_id);

    i++;
  }
  H5Fclose(link_file_id);
  
#ifdef LEGIONIO_PHASER_TIMERS
  // Create phase barriers used for timing info 
  this->pb_write = this->runtime->create_phase_barrier(ctx, i);
  this->pb_timer = this->runtime->create_phase_barrier(ctx, 1); 
  this->pb_read = this->runtime->create_phase_barrier(ctx, i);
#endif
}


// this version is used for unstructured index spaces 
void PersistentRegion::create_persistent_subregions(Context ctx,
    const char *name, LogicalRegion parent_lr, LogicalPartition lp,
    Coloring hdf_parts, std::map<FieldID, std::string> &field_map)
{
  hid_t link_file_id, shard_group_id, shard_ds_id, dataspace_id, dtype_id, shard_file_id, attr_ds_id, link_group_id, link_group_2_id;
  herr_t status;
#ifdef LEGIONIO_SETUP_LINKS 
  link_file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
#endif
  dtype_id  = H5Tcopy (H5T_NATIVE_DOUBLE);

  this->lp = lp;
  this->parent_lr = parent_lr;
  this->field_map = field_map;
  Rect<1> launch_rect(Point<1>(0), Point<1>(hdf_parts.size()-1));
  this->dom = Domain::from_rect<1>(launch_rect);
  
#ifdef LEGIONIO_SERIALIZE
  std::map<FieldID, std::string> field_map_des;
  Realm::Serialization::DynamicBufferSerializer dbs(0);
  dbs << field_map;
  this->field_map_size = dbs.bytes_used();
  this->field_map_serial = dbs.detach_buffer();
#endif
  //  Realm::Serialization::FixedBufferDeserializer fbd(this->field_map_serial, this->field_map_size);
  //  bool ok = fbd >> field_map_des;
  //  if(ok) { 
  //    std::cout << " In create_persistent_subregions and I deserialized!: " <<
  //      ok << std::endl;
  //  }

  int i = 0;
  int x_min_current = 0;
  
  for(unsigned int color = 0; color < hdf_parts.size(); color++){
    pieces.push_back(Piece());
    pieces[i].color = color;
    pieces[i].child_lr = runtime->get_logical_subregion_by_color(ctx, lp, color);
    pieces[i].parent_lr = parent_lr;
    FieldSpace fs = pieces[i].child_lr.get_field_space();
    pieces[i].dp = DomainPoint();
    pieces[i].dp.dim = 1; 
    int x_min  = x_min_current;
    pieces[i].dp[0] = i; 
    x_min_current  += hdf_parts[color].points.size();
    //pieces[i].dp[1] = x_min_current;
    int x_max = x_min_current;
    x_min_current ++; 
      
#ifdef LEGIONIO_VERBOSE
    std::cout << "Found a logical region (unstructured index space):  with color  " << color <<  std::endl;
#endif
    int *shard_dims;
    std::ostringstream ds_name_stream;
    
    ds_name_stream <<  pieces[i].color; 
    sprintf(pieces[i].shard_name, "%d-%d-%s",
            x_min, x_max, name); 
    
#ifdef LEGIONIO_VERBOSE 
    std::cout << "domain (unstructured) is: [" << x_min << "," << x_max
              << "]" << std::endl; 
#endif
    
    hsize_t dims[1];
    dims[0] = x_max+1;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    
    dims[0] = 2;
    attr_ds_id = H5Screate_simple(1, dims, NULL);
    
    shard_dims = (int*) malloc(2*sizeof(int)); 
    shard_dims[0] = x_min;
    shard_dims[1] = x_max;
    
    shard_file_id = H5Fcreate(pieces[i].shard_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    for (std::map<FieldID, std::string>::const_iterator iterator = field_map.begin();
         iterator != field_map.end(); iterator++) {
      FieldID fid = iterator->first;
      
      char* ds;
      char* gp;
      split_path_file(&gp, &ds, iterator->second.c_str());
#ifdef LEGIONIO_VERBOSE 
      std::cout << "creating shard file with group name:" << gp
                << " and dataset name " << ds << std::endl; 
#endif

    
      size_t field_size = runtime->get_field_size(ctx, fs, fid);
      status = H5Tset_size(dtype_id, field_size);
      
      if(H5Lexists(shard_file_id, gp, H5P_DEFAULT)) { 
        shard_group_id = H5Gopen2(shard_file_id, gp, H5P_DEFAULT);
      } else { 
        shard_group_id = H5Gcreate2(shard_file_id, gp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      } 
      
      if(H5Lexists(shard_group_id, ds, H5P_DEFAULT)) { 
        shard_ds_id = H5Dopen2(shard_group_id, ds, H5P_DEFAULT); 
      } else { 
        shard_ds_id = H5Dcreate2(shard_group_id, ds, H5T_NATIVE_DOUBLE,
                                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT);
      }
#ifdef LEGIONIO_SETUP_LINKS
      if(H5Lexists(link_file_id, gp, H5P_DEFAULT)) {
#ifdef LEGIONIO_VERBOSE
        std::cout << "Group " << gp << " exists in link file " << std::endl; 
#endif
        link_group_id = H5Gopen2(link_file_id, gp, H5P_DEFAULT);
      } else { 
#ifdef LEGIONIO_VERBOSE
        std::cout << "Creating group in link file " << gp << std::endl;
#endif
        link_group_id = H5Gcreate2(link_file_id, gp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      }
      if(H5Lexists(link_file_id, iterator->second.c_str(), H5P_DEFAULT)){
        link_group_2_id = H5Gopen2(link_group_id, ds, H5P_DEFAULT);
      } else {
#ifdef LEGIONIO_VERBOSE
        std::cout << "Creating another group in the link file " << ds << std::endl;
#endif
        link_group_2_id = H5Gcreate2(link_group_id, ds, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
      }
      H5Gclose(link_group_id);
      
      hid_t attr_id = H5Acreate2(shard_ds_id, "dims", H5T_NATIVE_INT, attr_ds_id,
                                 H5P_DEFAULT, H5P_DEFAULT);
      
      status = H5Awrite(attr_id, H5T_NATIVE_INT, shard_dims);
      H5Aclose(attr_id);
      
      H5Dclose(shard_ds_id);
      H5Gclose(shard_group_id);
      H5Fclose(shard_file_id);
      
      status = H5Lcreate_external(pieces[i].shard_name, iterator->second.c_str(),
                                  link_group_2_id, ds_name_stream.str().c_str(),
                                  H5P_DEFAULT, H5P_DEFAULT);
      
      shard_file_id = H5Fopen(pieces[i].shard_name, H5F_ACC_RDWR, H5P_DEFAULT);      
      H5Gclose(link_group_2_id);
#endif
    }
    H5Fclose(shard_file_id);
    
    i++;
  }
#ifdef LEGIONIO_SETUP_LINKS 
  H5Fclose(link_file_id);
#endif
  
#ifdef LEGIONIO_PHASER_TIMERS
  // Create phase barriers used for timing info 
  this->pb_write = this->runtime->create_phase_barrier(ctx, i);
  this->pb_timer = this->runtime->create_phase_barrier(ctx, 1); 
  this->pb_read = this->runtime->create_phase_barrier(ctx, i);
#endif
}


void split_path_file(char** p, char** f, const char *pf) {
    char *slash = (char*)pf, *next;
    while ((next = strpbrk(slash + 1, "\\/"))) slash = next;
    if (pf != slash) slash++;
    *p = strndup(pf, slash - pf);
    *f = strdup(slash);
}
