# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from object_msgs/trajectory_array.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import object_msgs.msg
import std_msgs.msg

class trajectory_array(genpy.Message):
  _md5sum = "92ff43ffdd6900a1509a628214083282"
  _type = "object_msgs/trajectory_array"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """trajectory[] all_car
int8 host_car_id
================================================================================
MSG: object_msgs/trajectory
std_msgs/Header header
int8 id                 # car id
bool[] mask             # validity : 1 or 0
int8 valid_sequence_num # ex) valid_sequence_num=4 means that only first 4 sequences are valid
Object[] trajectory     # 
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

================================================================================
MSG: object_msgs/Object
std_msgs/Header header
uint32 id

# The type of classification given to this object.
uint8 classification
uint8 CLASSIFICATION_UNKNOWN=0
uint8 CLASSIFICATION_CAR=1
uint8 CLASSIFICATION_PEDESTRIAN=2
uint8 CLASSIFICATION_CYCLIST=3

# The detected position and orientation of the object.
float32 x       # m
float32 y       # m
float32 yaw     # rad

float32 v       # m/s
float32 yawrate # rad/s

float32 a      # m/ss
float32 delta  # radian

# size
float32 L     # m
float32 W     # m
"""
  __slots__ = ['all_car','host_car_id']
  _slot_types = ['object_msgs/trajectory[]','int8']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       all_car,host_car_id

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(trajectory_array, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.all_car is None:
        self.all_car = []
      if self.host_car_id is None:
        self.host_car_id = 0
    else:
      self.all_car = []
      self.host_car_id = 0

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      length = len(self.all_car)
      buff.write(_struct_I.pack(length))
      for val1 in self.all_car:
        _v1 = val1.header
        _x = _v1.seq
        buff.write(_get_struct_I().pack(_x))
        _v2 = _v1.stamp
        _x = _v2
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v1.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = val1.id
        buff.write(_get_struct_b().pack(_x))
        length = len(val1.mask)
        buff.write(_struct_I.pack(length))
        pattern = '<%sB'%length
        buff.write(struct.Struct(pattern).pack(*val1.mask))
        _x = val1.valid_sequence_num
        buff.write(_get_struct_b().pack(_x))
        length = len(val1.trajectory)
        buff.write(_struct_I.pack(length))
        for val2 in val1.trajectory:
          _v3 = val2.header
          _x = _v3.seq
          buff.write(_get_struct_I().pack(_x))
          _v4 = _v3.stamp
          _x = _v4
          buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
          _x = _v3.frame_id
          length = len(_x)
          if python3 or type(_x) == unicode:
            _x = _x.encode('utf-8')
            length = len(_x)
          buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
          _x = val2
          buff.write(_get_struct_IB9f().pack(_x.id, _x.classification, _x.x, _x.y, _x.yaw, _x.v, _x.yawrate, _x.a, _x.delta, _x.L, _x.W))
      _x = self.host_car_id
      buff.write(_get_struct_b().pack(_x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.all_car is None:
        self.all_car = None
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.all_car = []
      for i in range(0, length):
        val1 = object_msgs.msg.trajectory()
        _v5 = val1.header
        start = end
        end += 4
        (_v5.seq,) = _get_struct_I().unpack(str[start:end])
        _v6 = _v5.stamp
        _x = _v6
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v5.frame_id = str[start:end].decode('utf-8', 'rosmsg')
        else:
          _v5.frame_id = str[start:end]
        start = end
        end += 1
        (val1.id,) = _get_struct_b().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        pattern = '<%sB'%length
        start = end
        s = struct.Struct(pattern)
        end += s.size
        val1.mask = s.unpack(str[start:end])
        val1.mask = list(map(bool, val1.mask))
        start = end
        end += 1
        (val1.valid_sequence_num,) = _get_struct_b().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.trajectory = []
        for i in range(0, length):
          val2 = object_msgs.msg.Object()
          _v7 = val2.header
          start = end
          end += 4
          (_v7.seq,) = _get_struct_I().unpack(str[start:end])
          _v8 = _v7.stamp
          _x = _v8
          start = end
          end += 8
          (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          start = end
          end += length
          if python3:
            _v7.frame_id = str[start:end].decode('utf-8', 'rosmsg')
          else:
            _v7.frame_id = str[start:end]
          _x = val2
          start = end
          end += 41
          (_x.id, _x.classification, _x.x, _x.y, _x.yaw, _x.v, _x.yawrate, _x.a, _x.delta, _x.L, _x.W,) = _get_struct_IB9f().unpack(str[start:end])
          val1.trajectory.append(val2)
        self.all_car.append(val1)
      start = end
      end += 1
      (self.host_car_id,) = _get_struct_b().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      length = len(self.all_car)
      buff.write(_struct_I.pack(length))
      for val1 in self.all_car:
        _v9 = val1.header
        _x = _v9.seq
        buff.write(_get_struct_I().pack(_x))
        _v10 = _v9.stamp
        _x = _v10
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v9.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = val1.id
        buff.write(_get_struct_b().pack(_x))
        length = len(val1.mask)
        buff.write(_struct_I.pack(length))
        pattern = '<%sB'%length
        buff.write(val1.mask.tostring())
        _x = val1.valid_sequence_num
        buff.write(_get_struct_b().pack(_x))
        length = len(val1.trajectory)
        buff.write(_struct_I.pack(length))
        for val2 in val1.trajectory:
          _v11 = val2.header
          _x = _v11.seq
          buff.write(_get_struct_I().pack(_x))
          _v12 = _v11.stamp
          _x = _v12
          buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
          _x = _v11.frame_id
          length = len(_x)
          if python3 or type(_x) == unicode:
            _x = _x.encode('utf-8')
            length = len(_x)
          buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
          _x = val2
          buff.write(_get_struct_IB9f().pack(_x.id, _x.classification, _x.x, _x.y, _x.yaw, _x.v, _x.yawrate, _x.a, _x.delta, _x.L, _x.W))
      _x = self.host_car_id
      buff.write(_get_struct_b().pack(_x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.all_car is None:
        self.all_car = None
      end = 0
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.all_car = []
      for i in range(0, length):
        val1 = object_msgs.msg.trajectory()
        _v13 = val1.header
        start = end
        end += 4
        (_v13.seq,) = _get_struct_I().unpack(str[start:end])
        _v14 = _v13.stamp
        _x = _v14
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v13.frame_id = str[start:end].decode('utf-8', 'rosmsg')
        else:
          _v13.frame_id = str[start:end]
        start = end
        end += 1
        (val1.id,) = _get_struct_b().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        pattern = '<%sB'%length
        start = end
        s = struct.Struct(pattern)
        end += s.size
        val1.mask = numpy.frombuffer(str[start:end], dtype=numpy.bool, count=length)
        val1.mask = list(map(bool, val1.mask))
        start = end
        end += 1
        (val1.valid_sequence_num,) = _get_struct_b().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.trajectory = []
        for i in range(0, length):
          val2 = object_msgs.msg.Object()
          _v15 = val2.header
          start = end
          end += 4
          (_v15.seq,) = _get_struct_I().unpack(str[start:end])
          _v16 = _v15.stamp
          _x = _v16
          start = end
          end += 8
          (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
          start = end
          end += 4
          (length,) = _struct_I.unpack(str[start:end])
          start = end
          end += length
          if python3:
            _v15.frame_id = str[start:end].decode('utf-8', 'rosmsg')
          else:
            _v15.frame_id = str[start:end]
          _x = val2
          start = end
          end += 41
          (_x.id, _x.classification, _x.x, _x.y, _x.yaw, _x.v, _x.yawrate, _x.a, _x.delta, _x.L, _x.W,) = _get_struct_IB9f().unpack(str[start:end])
          val1.trajectory.append(val2)
        self.all_car.append(val1)
      start = end
      end += 1
      (self.host_car_id,) = _get_struct_b().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_2I = None
def _get_struct_2I():
    global _struct_2I
    if _struct_2I is None:
        _struct_2I = struct.Struct("<2I")
    return _struct_2I
_struct_IB9f = None
def _get_struct_IB9f():
    global _struct_IB9f
    if _struct_IB9f is None:
        _struct_IB9f = struct.Struct("<IB9f")
    return _struct_IB9f
_struct_b = None
def _get_struct_b():
    global _struct_b
    if _struct_b is None:
        _struct_b = struct.Struct("<b")
    return _struct_b
