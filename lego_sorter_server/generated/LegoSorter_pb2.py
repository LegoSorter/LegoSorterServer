# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: LegoSorter.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import lego_sorter_server.generated.Messages_pb2 as Messages__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='LegoSorter.proto',
  package='sorter',
  syntax='proto3',
  serialized_options=b'\n\022com.lsorter.sorterB\017LegoSorterProto',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10LegoSorter.proto\x12\x06sorter\x1a\x0eMessages.proto\"\x15\n\x13SorterConfiguration2\xe4\x01\n\nLegoSorter\x12\x45\n\x10processNextImage\x12\x14.common.ImageRequest\x1a\x1b.common.ListOfBoundingBoxes\x12>\n\x10getConfiguration\x12\r.common.Empty\x1a\x1b.sorter.SorterConfiguration\x12O\n\x13updateConfiguration\x12\x1b.sorter.SorterConfiguration\x1a\x1b.sorter.SorterConfigurationB%\n\x12\x63om.lsorter.sorterB\x0fLegoSorterProtob\x06proto3'
  ,
  dependencies=[Messages__pb2.DESCRIPTOR,])




_SORTERCONFIGURATION = _descriptor.Descriptor(
  name='SorterConfiguration',
  full_name='sorter.SorterConfiguration',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=44,
  serialized_end=65,
)

DESCRIPTOR.message_types_by_name['SorterConfiguration'] = _SORTERCONFIGURATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SorterConfiguration = _reflection.GeneratedProtocolMessageType('SorterConfiguration', (_message.Message,), {
  'DESCRIPTOR' : _SORTERCONFIGURATION,
  '__module__' : 'LegoSorter_pb2'
  # @@protoc_insertion_point(class_scope:sorter.SorterConfiguration)
  })
_sym_db.RegisterMessage(SorterConfiguration)


DESCRIPTOR._options = None

_LEGOSORTER = _descriptor.ServiceDescriptor(
  name='LegoSorter',
  full_name='sorter.LegoSorter',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=68,
  serialized_end=296,
  methods=[
  _descriptor.MethodDescriptor(
    name='processNextImage',
    full_name='sorter.LegoSorter.processNextImage',
    index=0,
    containing_service=None,
    input_type=Messages__pb2._IMAGEREQUEST,
    output_type=Messages__pb2._LISTOFBOUNDINGBOXES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='getConfiguration',
    full_name='sorter.LegoSorter.getConfiguration',
    index=1,
    containing_service=None,
    input_type=Messages__pb2._EMPTY,
    output_type=_SORTERCONFIGURATION,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='updateConfiguration',
    full_name='sorter.LegoSorter.updateConfiguration',
    index=2,
    containing_service=None,
    input_type=_SORTERCONFIGURATION,
    output_type=_SORTERCONFIGURATION,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_LEGOSORTER)

DESCRIPTOR.services_by_name['LegoSorter'] = _LEGOSORTER

# @@protoc_insertion_point(module_scope)
