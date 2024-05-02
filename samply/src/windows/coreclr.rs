use std::{
    collections::{hash_map::Entry, HashMap, HashSet, VecDeque},
    convert::TryInto,
    fs::File,
    io::BufWriter,
    path::Path,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};

use bitflags::bitflags;
use debugid::DebugId;
use fxprof_processed_profile::{
    debugid, CategoryColor, CategoryHandle, CategoryPairHandle, CounterHandle, CpuDelta,
    FrameFlags, FrameInfo, LibraryHandle, LibraryInfo, MarkerDynamicField, MarkerFieldFormat,
    MarkerLocation, MarkerSchema, MarkerSchemaField, MarkerTiming, ProcessHandle, Profile,
    ProfilerMarker, ReferenceTimestamp, SamplingInterval, Symbol, SymbolTable, ThreadHandle,
    Timestamp,
};
use serde_json::{json, to_writer, Value};
use uuid::Uuid;

use crate::shared::context_switch::{
    ContextSwitchHandler, OffCpuSampleGroup, ThreadContextSwitchData,
};
use crate::shared::jit_category_manager::JitCategoryManager;
use crate::shared::lib_mappings::LibMappingInfo;
use crate::shared::lib_mappings::{LibMappingAdd, LibMappingOp, LibMappingOpQueue};
use crate::shared::process_sample_data::{MarkerSpanOnThread, ProcessSampleData, SimpleMarker};
use crate::shared::types::{StackFrame, StackMode};

use crate::shared::{
    jit_function_add_marker::JitFunctionAddMarker, marker_file::get_markers,
    process_sample_data::UserTimingMarker, timestamp_converter::TimestampConverter,
};

use super::etw_reader::{self, etw_types::EventRecord, schema::TypedEvent};
use super::etw_reader::{
    event_properties_to_string, open_trace,
    parser::{Address, Parser, TryParse},
    print_property,
    schema::SchemaLocator,
    write_property, GUID,
};
use super::*;

use super::ProfileContext;

bitflags! {
    #[derive(PartialEq, Eq)]
    pub struct CoreClrMethodFlagsMap: u32 {
        const dynamic = 0x1;
        const generic = 0x2;
        const has_shared_generic_code = 0x4;
        const jitted = 0x8;
        const jit_helper = 0x10;
        const profiler_rejected_precompiled_code = 0x20;
        const ready_to_run_rejected_precompiled_code = 0x40;

        // next three bits are the tiered compilation level
        const opttier_bit0 = 0x80;
        const opttier_bit1 = 0x100;
        const opttier_bit2 = 0x200;

        // extent flags/value (hot/cold)
        const extent_bit_0 = 0x10000000; // 0x1 == cold, 0x0 = hot
        const extent_bit_1 = 0x20000000; // always 0 for now looks like
        const extent_bit_2 = 0x40000000;
        const extent_bit_3 = 0x80000000;

        const _ = !0;
    }
    #[derive(PartialEq, Eq)]
    pub struct TieredCompilationSettingsMap: u32 {
        const None = 0x0;
        const QuickJit = 0x1;
        const QuickJitForLoops = 0x2;
        const TieredPGO = 0x4;
        const ReadyToRun = 0x8;
    }
}

pub fn handle_coreclr_event(context: &mut ProfileContext, s: &TypedEvent, parser: &mut Parser, timestamp_converter: &TimestampConverter) {
    let timestamp_raw = s.timestamp() as u64;
    let timestamp = timestamp_converter.convert_time(timestamp_raw);

    let Some(dotnet_event) = s.name().strip_prefix("Microsoft-Windows-DotNETRuntime/") else { panic!("Unexpected event {}", s.name()) };


    // TODO -- we may need to use the rundown provider if we trace running processes
    // https://learn.microsoft.com/en-us/dotnet/framework/performance/clr-etw-providers

    // We get DbgID_RSDS for ReadyToRun loaded images, along with PDB files. We also get ModuleLoad events for the same:
    // this means we can ignore the ModuleLoadEvents because we'll get dbginfo already mapped properly when the image
    // is loaded.
    //
    // DbgID_RSDS pid: 8104 0x7ffc48fa0000 System.Console.dll 98b8e660-83e9-14c1-b8b2-d24515644b95-1 System.Console.ni.pdb 1
    // Microsoft-Windows-DotNETRuntime/CLRLoader/ModuleLoad   ModuleID= 0x7ffb550f0eb0, ModuleILPath= C:\Program Files\dotnet\shared\Microsoft.NETCore.App\8.0.4\System.Console.dll, ModuleNativePath= ,   ClrInstanceID= 9,
    // ManagedPdbSignature= DE92BFC5-E16C-9E95-9454-13C5F6EA6389,   ManagedPdbAge= 1,   ManagedPdbBuildPath= ..., 
    //  NativePdbSignature= 98B8E660-83E9-14C1-B8B2-D24515644B95,   NativePdbAge= 1,   NativePdbBuildPath= System.Console.ni.pdb,
    let mut handled = false;

    if let Some(method_event) = dotnet_event.strip_prefix("CLRMethod/") {
        match method_event {
            "MethodJittingStarted" => {
                // I think we migth need this if we don't get MethodLoadVerbose events,
                // in order to connect the method name to the MethodLoad when jit is complete.

                //let method_id: u64 = parser.parse("MethodID");
                //let clr_instance_id: u32 = parser.parse("ClrInstanceID");
                //let method_basename: String = parser.parse("MethodName");
                //let method_namespace: String = parser.parse("MethodNamespace");
                //let method_signature: String = parser.parse("MethodSignature");
            }
            // there's MethodDCStart & MethodDCStartVerbose & MethodLoad
            // difference between *Verbose and not, is Verbose includes the names

            "MethodLoadVerbose"
            // | "R2RGetEntryPoint"
            => {
                // R2RGetEntryPoint shares a lot of fields with MethodLoadVerbose
                let is_r2r = method_event == "R2RGetEntryPoint";

                let process_id = s.process_id();
                context.ensure_process_jit_info(process_id);
                let Some(process) = context.get_process(process_id) else { return; };

                //let method_id: u64 = parser.parse("MethodID");
                //let clr_instance_id: u32 = parser.parse("ClrInstanceID"); // v1/v2 only

                let method_basename: String = parser.parse("MethodName");
                let method_namespace: String = parser.parse("MethodNamespace");
                let method_signature: String = parser.parse("MethodSignature");

                let method_start_address: Address = if is_r2r { parser.parse("EntryPoint") } else { parser.parse("MethodStartAddress") };
                let method_size: u32 = parser.parse("MethodSize"); // TODO: R2R doesn't have a size?

                // There's a v0, v1, and v2 version of this event. There are rules in `eventtrace.cpp` in the runtime
                // that describe the rules, but basically:
                // - during a first-JIT, only a v1 (not v0 and not v2+) MethodLoad is emitted.
                // - during a re-jit, a v2 event is emitted.
                // - v2 contains a "NativeCodeId" field which will be nonzero in v2. 
                // - the unique key for a method extent is MethodId + MethodCodeId + extent (hot/cold)

                // there's some stuff in MethodFlags -- might be tiered JIT info?
                // also ClrInstanceID -- we probably won't have more than one runtime, but maybe.

                let method_name = format!("{method_basename} [{method_namespace}] \u{2329}{method_signature}\u{232a}");

                let mut process_jit_info = context.get_process_jit_info(process_id);
                let start_address = method_start_address.as_u64();
                let relative_address = process_jit_info.next_relative_address;
                process_jit_info.next_relative_address += method_size;

                if let Some(main_thread) = process.main_thread_handle {
                    context.profile.borrow_mut().add_marker(
                        main_thread,
                        CategoryHandle::OTHER,
                        "JitFunctionAdd",
                        JitFunctionAddMarker(method_name.to_owned()),
                        MarkerTiming::Instant(timestamp),
                    );
                }

                let category = context.coreclr_category;
                let info = LibMappingInfo::new_jit_function(process_jit_info.lib_handle, category, None);
                process_jit_info.jit_mapping_ops.push(timestamp_raw, LibMappingOp::Add(LibMappingAdd {
                    start_avma: start_address,
                    end_avma: start_address + (method_size as u64),
                    relative_address_at_start: relative_address,
                    info
                }));
                process_jit_info.symbols.push(Symbol {
                    address: relative_address,
                    size: Some(method_size as u32),
                    name: method_name,
                });

                handled = true;
            }
            "ModuleLoad" | "ModuleDCStart" |
            "ModuleUnload" | "ModuleDCEnd" => {
                // do we need this for ReadyToRun code?

                //let module_id: u64 = parser.parse("ModuleID");
                //let assembly_id: u64 = parser.parse("AssemblyId");
                //let managed_pdb_signature: u?? = parser.parse("ManagedPdbSignature");
                //let managed_pdb_age: u?? = parser.parse("ManagedPdbAge");
                //let managed_pdb_path: String = parser.parse("ManagedPdbPath");
                //let native_pdb_signature: u?? = parser.parse("NativePdbSignature");
                //let native_pdb_age: u?? = parser.parse("NativePdbAge");
                //let native_pdb_path: String = parser.parse("NativePdbPath");
                handled = true;
            }
            _ => {
                //let text = event_properties_to_string(&s, &mut parser, None);
                //eprintln!("Method event: {dotnet_event} {text}");
            }
        }
    } else if dotnet_event == "Type/BulkType" {
        //         <template tid="BulkType">
        // <data name="Count" inType="win:UInt32"    />
        // <data name="ClrInstanceID" inType="win:UInt16" />
        // <struct name="Values" count="Count" >
          // <data name="TypeID" inType="win:UInt64" outType="win:HexInt64" />
          // <data name="ModuleID" inType="win:UInt64" outType="win:HexInt64" />
          // <data name="TypeNameID" inType="win:UInt32" />
          // <data name="Flags" inType="win:UInt32" map="TypeFlagsMap"/>
          // <data name="CorElementType"  inType="win:UInt8" />
          // <data name="Name" inType="win:UnicodeString" />
          // <data name="TypeParameterCount" inType="win:UInt32" />
          // <data name="TypeParameters"  count="TypeParameterCount"  inType="win:UInt64" outType="win:HexInt64" />
        // </struct>
        // <UserData>
          // <Type xmlns="myNs">
            // <Count> %1 </Count>
            // <ClrInstanceID> %2 </ClrInstanceID>
          // </Type>
        // </UserData>
        //let count: u32 = parser.parse("Count");

        // uint32 + uint16 at the front (Count and ClrInstanceID), then struct of values. We don't need a Vec<u8> copy.
        //let values: Vec<u8> = parser.parse("Values");
        //let values = &s.user_buffer()[6..];

        //eprintln!("Type/BulkType count: {} user_buffer size: {} values len: {}", count, s.user_buffer().len(), values.len());
    } else if dotnet_event == "CLRStack/CLRStackWalk" {
        let process_id = s.process_id();

        //let text = event_properties_to_string(s, parser, None);
        //eprintln!("{dotnet_event} {text}");
        handled = true;
    }
    
    if !handled {
        if dotnet_event.contains("GarbageCollection") { return }
        if dotnet_event.contains("/Thread") { return }
        if dotnet_event.contains("Type/BulkType") { return }
        let text = event_properties_to_string(s, parser, None);
        //eprintln!("Unhandled .NET event: {dotnet_event} {text}");
    }
}