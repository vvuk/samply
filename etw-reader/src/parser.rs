//! ETW Types Parser
//!
//! This module act as a helper to parse the Buffer from an ETW Event
use crate::etw_types::EVENT_HEADER_FLAG_32_BIT_HEADER;
use crate::sddl;
use crate::tdh;
use crate::tdh_types::{Property, PropertyFlags, TdhInType, TdhOutType};
use crate::property::{PropertyInfo, PropertyIter};
use crate::schema::TypedEvent;
use crate::utils;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::convert::TryInto;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::rc::Rc;
use windows::Guid;

/// Parser module errors
#[derive(Debug)]
pub enum ParserError {
    /// An invalid type...
    InvalidType,
    /// Error parsing
    ParseError,
    /// Length mismatch when parsing a type
    LengthMismatch,
    PropertyError(String),
    /// An error while transforming an Utf-8 buffer into String
    Utf8Error(std::string::FromUtf8Error),
    /// An error trying to get an slice as an array
    SliceError(std::array::TryFromSliceError),
    /// Represents an internal [SddlNativeError]
    ///
    /// [SddlNativeError]: sddl::SddlNativeError
    //SddlNativeError(sddl::SddlNativeError),
    /// Represents an internal [TdhNativeError]
    ///
    /// [TdhNativeError]: tdh::TdhNativeError
    TdhNativeError(tdh::TdhNativeError),
}

impl From<tdh::TdhNativeError> for ParserError {
    fn from(err: tdh::TdhNativeError) -> Self {
        ParserError::TdhNativeError(err)
    }
}
/*
impl From<sddl::SddlNativeError> for ParserError {
    fn from(err: sddl::SddlNativeError) -> Self {
        ParserError::SddlNativeError(err)
    }
}*/

impl From<std::string::FromUtf8Error> for ParserError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        ParserError::Utf8Error(err)
    }
}

impl From<std::array::TryFromSliceError> for ParserError {
    fn from(err: std::array::TryFromSliceError) -> Self {
        ParserError::SliceError(err)
    }
}

type ParserResult<T> = Result<T, ParserError>;

/// Trait to try and parse a type
///
/// This trait has to be implemented in order to be able to parse a type we want to retrieve from
/// within an Event. On success the parsed value will be returned within a Result, on error an Err
/// should be returned accordingly
///
/// An implementation for most of the Primitive Types is created by using a Macro, any other needed type
/// requires this trait to be implemented
// TODO: Find a way to use turbofish operator
pub trait TryParse<T> {
    /// Implement the `try_parse` function to provide a way to Parse `T` from an ETW event or
    /// return an Error in case the type `T` can't be parsed
    ///
    /// # Arguments
    /// * `name` - Name of the property to be found in the Schema
    fn try_parse(&mut self, name: &str) -> Result<T, ParserError>;
    fn parse(&mut self, name: &str) -> T {
        self.try_parse(name).unwrap()
    }
}

/// Represents a Parser
///
/// This structure holds the necessary data to parse the ETW event and retrieve the data from the
/// event
#[allow(dead_code)]
pub struct Parser<'a> {
    event: &'a TypedEvent<'a>,
    properties: &'a PropertyIter,
    pub buffer: &'a [u8],
    last_property: u32,
    offset: usize,
    // a map from property indx to PropertyInfo
    cache: Vec<Rc<PropertyInfo>>,
}

impl<'a> Parser<'a> {
    /// Use the `create` function to create an instance of a Parser
    ///
    /// # Arguments
    /// * `schema` - The [Schema] from the ETW Event we want to parse
    ///
    /// # Example
    /// ```rust
    /// let my_callback = |record: EventRecord, schema_locator: &mut SchemaLocator| {
    ///     let schema = schema_locator.event_schema(record)?;
    ///     let parser = Parse::create(&schema);
    /// };
    /// ```
    pub fn create(event: &'a TypedEvent) -> Self {
        Parser {
            event,
            buffer: event.user_buffer(),
            properties: event.schema.properties(),
            last_property: 0,
            offset: 0,
            cache: Vec::new(), // We could fill the cache on creation
        }
    }

    #[allow(dead_code)]
    fn fill_cache(
        schema: &TypedEvent,
        properties: &PropertyIter,
    ) -> ParserResult<HashMap<String, PropertyInfo>> {
        let user_buffer_len = schema.user_buffer().len();
        let mut prop_offset = 0;
        panic!();
/*
        Ok(properties.properties_iter().iter().try_fold(
            HashMap::new(),
            |mut cache, x| -> ParserResult<HashMap<String, PropertyInfo>> {
                let prop_size = tdh::property_size(schema.record(), &x.name)? as usize;

                if user_buffer_len < prop_size {
                    return Err(ParserError::PropertyError(
                        "Property length out of buffer bounds".to_owned(),
                    ));
                }
                let prop_buffer = schema.user_buffer()[..prop_size]
                    .iter()
                    .take(prop_size)
                    .cloned()
                    .collect();

                cache.insert(x.name.clone(), PropertyInfo::create(x.clone(), prop_offset, prop_buffer));
                prop_offset += prop_size;

                Ok(cache)
            },
        )?)*/
    }

    // TODO: Find a cleaner way to do this, not very happy with it rn
    fn find_property_size(&self, property: &Property) -> ParserResult<usize> {
        if property
            .flags
            .intersects(PropertyFlags::PROPERTY_PARAM_LENGTH)
            && property.len() > 0
        {
            let size;
            if property.in_type() == TdhInType::InTypePointer {
                size = if (self.event.event_flags() & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0 {
                    4
                } else {
                    8
                };
            } else {
                size = property.len() as usize;
            }
            return Ok(size);
        }
        // TODO: Study heuristic method used in krabsetw :)
        if property.flags.is_empty() && property.len() > 0 && property.count == 1 {
            return Ok(property.len());
        }
        if property.count == 1 {
            match property.in_type {
                TdhInType::InTypeInt32 | TdhInType::InTypeUInt32 => return Ok(4),
                TdhInType::InTypeInt64 | TdhInType::InTypeUInt64 => return Ok(8),
                TdhInType::InTypeInt8 | TdhInType::InTypeUInt8 => return Ok(1),
                TdhInType::InTypePointer => return Ok(if (self.event.event_flags() & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0 {
                    4
                } else {
                    8
                }),
                TdhInType::InTypeGuid => return Ok(std::mem::size_of::<Guid>()),
                TdhInType::InTypeUnicodeString => {
                    return Ok(utils::parse_unk_size_null_unicode_vec(&self.buffer).len()*2)
                }
                TdhInType::InTypeAnsiString => {
                    return Ok(utils::parse_unk_size_null_ansi_vec(&self.buffer).len())
                }
                _ => {}
            }
        }
        Ok(tdh::property_size(self.event.record(), &property.name).unwrap() as usize)
    }

    pub fn find_property(&mut self, name: &str) -> ParserResult<usize> {
        let indx = *self.properties.name_to_indx.get(name).ok_or_else(
            || ParserError::PropertyError("Unknown property".to_owned()))?;
        if indx < self.cache.len()  {
            return Ok(indx);
        }

        let mut prop_info = Rc::new(PropertyInfo::default());

        // TODO: Find a way to do this with an iter, try_find looks promising but is not stable yet
        // TODO: Clean this a bit, not a big fan of this loop
        for i in self.cache.len()..=indx {
            let curr_prop = self.properties.property(i).unwrap();

            let prop_size = self.find_property_size(&curr_prop)?;

            if self.buffer.len() < prop_size {
                return Err(ParserError::PropertyError(
                    "Property length out of buffer bounds".to_owned(),
                ));
            }

            // TODO: Evaluate not cloning the Property nor the buffer
            // We drain the buffer, if everything works correctly in the end the buffer will be empty
            // and we should have all properties in the cache
            let (prop_buffer, remaining) = self.buffer.split_at(prop_size);
            self.buffer = remaining;
            prop_info = Rc::from(PropertyInfo::create(curr_prop.clone(), self.offset, prop_buffer.to_owned()));
            self.cache.push(Rc::clone(&prop_info));
            self.offset += prop_size;
        }
        Ok(indx)
    }
}

macro_rules! impl_try_parse_primitive {
    ($T:ident, $ty:ident) => {
        impl TryParse<$T> for Parser<'_> {
            fn try_parse(&mut self, name: &str) -> ParserResult<$T> {
                use TdhInType::*;
                let indx = self.find_property(name)?;
                let prop_info = &self.cache[indx];
                let prop_info: &PropertyInfo = prop_info.borrow();

                if prop_info.property.in_type() != $ty {
                    return Err(ParserError::InvalidType)
                }
                if std::mem::size_of::<$T>() != prop_info.buffer.len() {
                    return Err(ParserError::LengthMismatch);
                }
                Ok($T::from_ne_bytes(prop_info.buffer.as_slice().try_into()?))
            }
        }
    };
}

impl_try_parse_primitive!(u8, InTypeUInt8);
impl_try_parse_primitive!(i8, InTypeInt8);
impl_try_parse_primitive!(u16, InTypeUInt16);
impl_try_parse_primitive!(i16, InTypeInt16);
impl_try_parse_primitive!(u32, InTypeUInt32);
impl_try_parse_primitive!(i32, InTypeInt32);
//impl_try_parse_primitive!(u64, InTypeUInt64);
impl_try_parse_primitive!(i64, InTypeInt64);

impl TryParse<u64> for Parser<'_> {
    fn try_parse(&mut self, name: &str) -> ParserResult<u64> {
        use TdhInType::*;
        let indx = self.find_property(name)?;
        let prop_info = &self.cache[indx];

        if prop_info.property.in_type() == InTypeUInt64 {
            if std::mem::size_of::<u64>() != prop_info.buffer.len() {
                return Err(ParserError::LengthMismatch);
            }
            return Ok(u64::from_ne_bytes(prop_info.buffer.as_slice().try_into()?));
        }
        if prop_info.property.in_type() == InTypePointer {
            if (self.event.event_flags() & EVENT_HEADER_FLAG_32_BIT_HEADER) != 0  {
                if std::mem::size_of::<u32>() != prop_info.buffer.len() {
                    return Err(ParserError::LengthMismatch);
                }
                return Ok(u32::from_ne_bytes(prop_info.buffer.as_slice().try_into()?) as u64);
            }
            if std::mem::size_of::<u64>() != prop_info.buffer.len() {
                return Err(ParserError::LengthMismatch);
            }
            return Ok(u64::from_ne_bytes(prop_info.buffer.as_slice().try_into()?));
        }
        return Err(ParserError::InvalidType)
    }
}

/// The `String` impl of the `TryParse` trait should be used to retrieve the following [TdhInTypes]:
///
/// * InTypeUnicodeString
/// * InTypeAnsiString
/// * InTypeCountedString
/// * InTypeGuid
///
/// On success a `String` with the with the data from the `name` property will be returned
///
/// # Arguments
/// * `name` - Name of the property to be found in the Schema

/// # Example
/// ```rust
/// let my_callback = |record: EventRecord, schema_locator: &mut SchemaLocator| {
///     let schema = schema_locator.event_schema(record)?;
///     let parser = Parse::create(&schema);
///     let image_name: String = parser.try_parse("ImageName")?;
/// };
/// ```
///
/// [TdhInTypes]: TdhInType
impl TryParse<String> for Parser<'_> {
    fn try_parse(&mut self, name: &str) -> ParserResult<String> {
        let indx = self.find_property(name)?;
        let prop_info = &self.cache[indx];
        let prop_info: &PropertyInfo = prop_info.borrow();

        // TODO: Handle errors and type checking better
        let res = match prop_info.property.in_type() {
            TdhInType::InTypeUnicodeString => {
                utils::parse_null_utf16_string(prop_info.buffer.as_slice())
            }
            TdhInType::InTypeAnsiString => String::from_utf8(prop_info.buffer.clone())?
                .trim_matches(char::default())
                .to_string(),
            TdhInType::InTypeSid => {
                panic!()
                //sddl::convert_sid_to_string(prop_info.buffer.as_ptr() as isize)?
            }
            TdhInType::InTypeCountedString => unimplemented!(),
            _ => return Err(ParserError::InvalidType),
        };

        Ok(res)
    }
}

impl TryParse<Guid> for Parser<'_> {
    fn try_parse(&mut self, name: &str) -> Result<Guid, ParserError> {
        let indx = self.find_property(name)?;
        let prop_info = &self.cache[indx];
        let prop_info: &PropertyInfo = prop_info.borrow();
        let res = match prop_info.property.in_type() {
            TdhInType::InTypeUnicodeString => {
                let guid_string = utils::parse_utf16_guid(prop_info.buffer.as_slice());

                if guid_string.len() != 36 {
                    return Err(ParserError::LengthMismatch);
                }

                return Ok(Guid::from(guid_string.as_str()))
            }
            TdhInType::InTypeGuid => {
                return Ok(Guid::from_values(u32::from_ne_bytes((&prop_info.buffer[0..4]).try_into()?),
                                            u16::from_ne_bytes((&prop_info.buffer[4..6]).try_into()?),
                                            u16::from_ne_bytes((&prop_info.buffer[6..8]).try_into()?),
                                            [
                                                prop_info.buffer[8],
                                                prop_info.buffer[9],
                                                prop_info.buffer[10],
                                                prop_info.buffer[11],
                                                prop_info.buffer[12],
                                                prop_info.buffer[13],
                                                prop_info.buffer[14],
                                                prop_info.buffer[15],
                                            ]
                                        ))
            }
            _ => return Err(ParserError::InvalidType),
        };
    }
}

impl TryParse<IpAddr> for Parser<'_> {
    fn try_parse(&mut self, name: &str) -> ParserResult<IpAddr> {
        let indx = self.find_property(name)?;
        let prop_info = &self.cache[indx];
        let prop_info: &PropertyInfo = prop_info.borrow();

        if prop_info.property.out_type() != TdhOutType::OutTypeIpv4
            && prop_info.property.out_type() != TdhOutType::OutTypeIpv6
        {
            return Err(ParserError::InvalidType);
        }

        // Hardcoded values for now
        let res = match prop_info.property.len() {
            16 => {
                let tmp: [u8; 16] = prop_info.buffer.as_slice().try_into()?;
                IpAddr::V6(Ipv6Addr::from(tmp))
            }
            4 => {
                let tmp: [u8; 4] = prop_info.buffer.as_slice().try_into()?;
                IpAddr::V4(Ipv4Addr::from(tmp))
            }
            _ => return Err(ParserError::LengthMismatch),
        };

        Ok(res)
    }
}

#[derive(Clone, Default, Debug)]
pub struct Pointer(usize);

impl std::ops::Deref for Pointer {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Pointer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl std::fmt::LowerHex for Pointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = self.0;

        std::fmt::LowerHex::fmt(&val, f) // delegate to u32/u64 implementation
    }
}

impl std::fmt::UpperHex for Pointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = self.0;

        std::fmt::UpperHex::fmt(&val, f) // delegate to u32/u64 implementation
    }
}

impl std::fmt::Display for Pointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let val = self.0;

        std::fmt::Display::fmt(&val, f) // delegate to u32/u64 implementation
    }
}

impl TryParse<Pointer> for Parser<'_> {
    fn try_parse(&mut self, name: &str) -> ParserResult<Pointer> {
        let indx = self.find_property(name)?;
        let prop_info = &self.cache[indx];
        let prop_info: &PropertyInfo = prop_info.borrow();

        let mut res = Pointer::default();
        if prop_info.buffer.len() == std::mem::size_of::<u32>() {
            res.0 = TryParse::<u32>::try_parse(self, name)? as usize;
        } else {
            res.0 = TryParse::<u64>::try_parse(self, name)? as usize;
        }

        Ok(res)
    }
}

impl TryParse<Vec<u8>> for Parser<'_> {
    fn try_parse(&mut self, name: &str) -> Result<Vec<u8>, ParserError> {
        let indx = self.find_property(name)?;
        let prop_info = &self.cache[indx];
        let prop_info: &PropertyInfo = prop_info.borrow();

        Ok(prop_info.buffer.clone())
    }
}

// TODO: Implement SocketAddress
// TODO: Study if we can use primitive types for HexInt64, HexInt32 and Pointer
