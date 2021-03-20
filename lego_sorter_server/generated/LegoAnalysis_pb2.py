# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: LegoAnalysis.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import lego_sorter_server.generated.Messages_pb2 as Messages__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='LegoAnalysis.proto',
  package='analysis',
  syntax='proto3',
  serialized_options=b'\n\024com.lsorter.analysisB\021LegoAnalysisProto',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x12LegoAnalysis.proto\x12\x08\x61nalysis\x1a\x0eMessages.proto2\x9f\x01\n\x0cLegoAnalysis\x12\x41\n\x0c\x44\x65tectBricks\x12\x14.common.ImageRequest\x1a\x1b.common.ListOfBoundingBoxes\x12L\n\x17\x44\x65tectAndClassifyBricks\x12\x14.common.ImageRequest\x1a\x1b.common.ListOfBoundingBoxesB)\n\x14\x63om.lsorter.analysisB\x11LegoAnalysisProtob\x06proto3'
  ,
  dependencies=[Messages__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)


DESCRIPTOR._options = None

_LEGOANALYSIS = _descriptor.ServiceDescriptor(
  name='LegoAnalysis',
  full_name='analysis.LegoAnalysis',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=49,
  serialized_end=208,
  methods=[
  _descriptor.MethodDescriptor(
    name='DetectBricks',
    full_name='analysis.LegoAnalysis.DetectBricks',
    index=0,
    containing_service=None,
    input_type=Messages__pb2._IMAGEREQUEST,
    output_type=Messages__pb2._LISTOFBOUNDINGBOXES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='DetectAndClassifyBricks',
    full_name='analysis.LegoAnalysis.DetectAndClassifyBricks',
    index=1,
    containing_service=None,
    input_type=Messages__pb2._IMAGEREQUEST,
    output_type=Messages__pb2._LISTOFBOUNDINGBOXES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_LEGOANALYSIS)

DESCRIPTOR.services_by_name['LegoAnalysis'] = _LEGOANALYSIS

# @@protoc_insertion_point(module_scope)