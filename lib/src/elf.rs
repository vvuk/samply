use crate::error::{GetSymbolsError, Result};
use crate::shared::{
    object_to_map, AddressDebugInfo, InlineStackFrame, SymbolicationQuery, SymbolicationResult,
};
use addr2line::{fallible_iterator, gimli, object};
use fallible_iterator::FallibleIterator;
use object::read::File;
use object::read::Object;
use object::SectionKind;
use std::cmp;
use uuid::Uuid;

const UUID_SIZE: usize = 16;
const PAGE_SIZE: usize = 4096;

pub fn get_symbolication_result<R>(buffer: &[u8], query: SymbolicationQuery) -> Result<R>
where
    R: SymbolicationResult,
{
    let SymbolicationQuery {
        breakpad_id,
        addresses,
        ..
    } = query;
    let elf_file = File::parse(buffer)
        .map_err(|_| GetSymbolsError::InvalidInputError("Could not parse ELF header"))?;
    let elf_id = get_elf_id(&elf_file)
        .ok_or_else(|| GetSymbolsError::InvalidInputError("id cannot be read"))?;
    let elf_id_string = format!("{:X}0", elf_id.to_simple());
    if elf_id_string != breakpad_id {
        return Err(GetSymbolsError::UnmatchedBreakpadId(
            elf_id_string,
            breakpad_id.to_string(),
        ));
    }
    let map = object_to_map(&elf_file);
    let mut symbolication_result = R::from_map(map, addresses);

    if R::wants_address_debug_info() {
        if let Ok(context) = addr2line::Context::new(&elf_file) {
            for address in addresses {
                if let Ok(frame_iter) = context.find_frames(*address as u64) {
                    let frames: std::result::Result<Vec<_>, _> =
                        frame_iter.map(convert_stack_frame).collect();
                    if let Ok(frames) = frames {
                        symbolication_result
                            .add_address_debug_info(*address, AddressDebugInfo { frames });
                    }
                }
            }
        }
    }

    Ok(symbolication_result)
}

fn convert_stack_frame<R: gimli::Reader>(
    frame: addr2line::Frame<R>,
) -> std::result::Result<InlineStackFrame, gimli::read::Error> {
    Ok(InlineStackFrame {
        function: frame
            .function
            .and_then(|f| f.demangle().ok().map(|n| n.into_owned())),
        file_path: frame
            .location
            .as_ref()
            .and_then(|l| l.file)
            .map(|f| f.to_owned()),
        line_number: frame.location.and_then(|l| l.line).map(|l| l as u32),
    })
}

fn create_elf_id(identifier: &[u8], little_endian: bool) -> Uuid {
    // Make sure that we have exactly UUID_SIZE bytes available
    let mut data = [0 as u8; UUID_SIZE];
    let len = cmp::min(identifier.len(), UUID_SIZE);
    data[0..len].copy_from_slice(&identifier[0..len]);

    if little_endian {
        // The file ELF file targets a little endian architecture. Convert to
        // network byte order (big endian) to match the Breakpad processor's
        // expectations. For big endian object files, this is not needed.
        data[0..4].reverse(); // uuid field 1
        data[4..6].reverse(); // uuid field 2
        data[6..8].reverse(); // uuid field 3
    }

    Uuid::from_bytes(data)
}

/// Tries to obtain the object identifier of an ELF object.
///
/// As opposed to Mach-O, ELF does not specify a unique ID for object files in
/// its header. Compilers and linkers usually add either `SHT_NOTE` sections or
/// `PT_NOTE` program header elements for this purpose. If one of these notes
/// is present, ElfFile's build_id() method will find it.
///
/// If neither of the above are present, this function will hash the first page
/// of the `.text` section (program code). This matches what the Breakpad
/// processor does.
///
/// If all of the above fails, this function will return `None`.
pub fn get_elf_id<'a>(elf_file: &File<'a>) -> Option<Uuid> {
    if let Some(identifier) = elf_file.build_id().ok()? {
        return Some(create_elf_id(identifier, elf_file.is_little_endian()));
    }

    // We were not able to locate the build ID, so fall back to hashing the
    // first page of the ".text" (program code) section. This algorithm XORs
    // 16-byte chunks directly into a UUID buffer.
    if let Some(section_data) = find_text_section(elf_file) {
        let mut hash = [0; UUID_SIZE];
        for i in 0..cmp::min(section_data.len(), PAGE_SIZE) {
            hash[i % UUID_SIZE] ^= section_data[i];
        }

        return Some(create_elf_id(&hash, elf_file.is_little_endian()));
    }

    None
}

/// Returns a reference to the data of the the .text section in an ELF binary.
fn find_text_section<'a>(file: &File<'a>) -> Option<&'a [u8]> {
    use object::read::ObjectSection;
    file.sections()
        .find(|header| header.kind() == SectionKind::Text)
        .and_then(|header| header.data().ok())
}
