"""Runtime-generated Upstox MarketDataFeedV3 protobuf classes.

Generated from the public Upstox MarketDataFeedV3.proto schema so the project
can decode the binary V3 feed without requiring protoc at runtime.
"""
from google.protobuf import descriptor_pb2, descriptor_pool, message
from google.protobuf import wrappers_pb2
from google.protobuf.internal import builder

_FDP = descriptor_pb2.FileDescriptorProto()
_FDP.name = "MarketDataFeedV3.proto"
_FDP.package = "com.upstox.marketdatafeederv3udapi.rpc.proto"
_FDP.syntax = "proto3"
_FDP.dependency.append("google/protobuf/wrappers.proto")


def field(msg, name, number, typ, label=1, type_name=None, oneof_index=None):
    f = msg.field.add(); f.name=name; f.number=number; f.type=typ; f.label=label
    if type_name: f.type_name=type_name
    if oneof_index is not None: f.oneof_index=oneof_index
    return f

def msg(name): return _FDP.message_type.add(name=name)

def enum(parent, name, values):
    e=parent.enum_type.add(name=name)
    for n,v in values: e.value.add(name=n, number=v)
    return e

LTPC=msg("LTPC")
field(LTPC,"ltp",1,1); field(LTPC,"ltt",2,3); field(LTPC,"ltq",3,3); field(LTPC,"cp",4,1)
field(LTPC,"iep",5,11,type_name=".google.protobuf.DoubleValue")
MarketLevel=msg("MarketLevel"); field(MarketLevel,"bidAskQuote",1,11,3,".com.upstox.marketdatafeederv3udapi.rpc.proto.Quote")
MarketOHLC=msg("MarketOHLC"); field(MarketOHLC,"ohlc",1,11,3,".com.upstox.marketdatafeederv3udapi.rpc.proto.OHLC")
Quote=msg("Quote"); field(Quote,"bidQ",1,3); field(Quote,"bidP",2,1); field(Quote,"askQ",3,3); field(Quote,"askP",4,1)
OptionGreeks=msg("OptionGreeks")
for i,n in enumerate(["delta","theta","gamma","vega","rho"],1): field(OptionGreeks,n,i,1)
OHLC=msg("OHLC"); field(OHLC,"interval",1,9); field(OHLC,"open",2,1); field(OHLC,"high",3,1); field(OHLC,"low",4,1); field(OHLC,"close",5,1); field(OHLC,"vol",6,3); field(OHLC,"ts",7,3)

enum(_FDP,"Type",[("initial_feed",0),("live_feed",1),("market_info",2)])
MarketFullFeed=msg("MarketFullFeed")
field(MarketFullFeed,"ltpc",1,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.LTPC")
field(MarketFullFeed,"marketLevel",2,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.MarketLevel")
field(MarketFullFeed,"optionGreeks",3,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.OptionGreeks")
field(MarketFullFeed,"marketOHLC",4,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.MarketOHLC")
for num,n,t in [(5,"atp",1),(6,"vtt",3),(7,"oi",1),(8,"iv",1),(9,"tbq",1),(10,"tsq",1),(11,"iep",1),(12,"rp",1),(13,"ieq",3),(14,"iiqTotal",3),(15,"iiqM",3),(16,"casEligible",8)]: field(MarketFullFeed,n,num,t)
IndexFullFeed=msg("IndexFullFeed"); field(IndexFullFeed,"ltpc",1,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.LTPC"); field(IndexFullFeed,"marketOHLC",2,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.MarketOHLC")
FullFeed=msg("FullFeed"); FullFeed.oneof_decl.add(name="FullFeedUnion"); field(FullFeed,"marketFF",1,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.MarketFullFeed",oneof_index=0); field(FullFeed,"indexFF",2,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.IndexFullFeed",oneof_index=0)
First=msg("FirstLevelWithGreeks"); field(First,"ltpc",1,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.LTPC"); field(First,"firstDepth",2,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.Quote"); field(First,"optionGreeks",3,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.OptionGreeks"); field(First,"vtt",4,3); field(First,"oi",5,1); field(First,"iv",6,1)

enum(_FDP,"RequestMode",[("ltpc",0),("full_d5",1),("option_greeks",2),("full_d30",3)])
Feed=msg("Feed"); Feed.oneof_decl.add(name="FeedUnion"); field(Feed,"ltpc",1,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.LTPC",oneof_index=0); field(Feed,"fullFeed",2,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.FullFeed",oneof_index=0); field(Feed,"firstLevelWithGreeks",3,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.FirstLevelWithGreeks",oneof_index=0); field(Feed,"requestMode",4,14,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.RequestMode")

enum(_FDP,"MarketStatus",[("PRE_OPEN_START",0),("PRE_OPEN_END",1),("NORMAL_OPEN",2),("NORMAL_CLOSE",3),("CLOSING_START",4),("CLOSING_END",5)])
Status=msg("StatusInfo"); field(Status,"status",1,9); field(Status,"updatedTime",2,3)
MarketInfo=msg("MarketInfo")
for name,typ in [("segmentStatus",".com.upstox.marketdatafeederv3udapi.rpc.proto.MarketStatus"),("casMarketStatus",".com.upstox.marketdatafeederv3udapi.rpc.proto.StatusInfo"),("preOpenSessionStatus",".com.upstox.marketdatafeederv3udapi.rpc.proto.StatusInfo")]:
    entry=descriptor_pb2.DescriptorProto(); entry.name=name.title().replace("Status", "Status")+"Entry"; entry.options.map_entry=True
    field(entry,"key",1,9); field(entry,"value",2,14 if "segment" in name else 11,type_name=typ if "segment" in name else typ)
    MarketInfo.nested_type.add().CopyFrom(entry); tn=f".com.upstox.marketdatafeederv3udapi.rpc.proto.MarketInfo.{entry.name}"; field(MarketInfo,name,1 if name=="segmentStatus" else 2 if name=="casMarketStatus" else 3,11,3,tn)
FeedResponse=msg("FeedResponse")
# map<string, Feed> feeds
entry=descriptor_pb2.DescriptorProto(name="FeedsEntry"); entry.options.map_entry=True; field(entry,"key",1,9); field(entry,"value",2,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.Feed"); FeedResponse.nested_type.add().CopyFrom(entry)
field(FeedResponse,"type",1,14,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.Type"); field(FeedResponse,"feeds",2,11,3,".com.upstox.marketdatafeederv3udapi.rpc.proto.FeedResponse.FeedsEntry"); field(FeedResponse,"currentTs",3,3); field(FeedResponse,"marketInfo",4,11,type_name=".com.upstox.marketdatafeederv3udapi.rpc.proto.MarketInfo")

_pool=descriptor_pool.Default()
_pool.AddSerializedFile(_FDP.SerializeToString())
_builder=builder.BuildMessageAndEnumDescriptors(_pool.FindFileByName(_FDP.name), globals())
builder.BuildTopDescriptorsAndMessages(_pool.FindFileByName(_FDP.name), __name__, globals())
