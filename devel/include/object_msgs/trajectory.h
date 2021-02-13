// Generated by gencpp from file object_msgs/trajectory.msg
// DO NOT EDIT!


#ifndef OBJECT_MSGS_MESSAGE_TRAJECTORY_H
#define OBJECT_MSGS_MESSAGE_TRAJECTORY_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <object_msgs/Object.h>

namespace object_msgs
{
template <class ContainerAllocator>
struct trajectory_
{
  typedef trajectory_<ContainerAllocator> Type;

  trajectory_()
    : header()
    , id(0)
    , mask()
    , valid_sequence_num(0)
    , trajectory()  {
    }
  trajectory_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , id(0)
    , mask(_alloc)
    , valid_sequence_num(0)
    , trajectory(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef int8_t _id_type;
  _id_type id;

   typedef std::vector<uint8_t, typename ContainerAllocator::template rebind<uint8_t>::other >  _mask_type;
  _mask_type mask;

   typedef int8_t _valid_sequence_num_type;
  _valid_sequence_num_type valid_sequence_num;

   typedef std::vector< ::object_msgs::Object_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::object_msgs::Object_<ContainerAllocator> >::other >  _trajectory_type;
  _trajectory_type trajectory;





  typedef boost::shared_ptr< ::object_msgs::trajectory_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::object_msgs::trajectory_<ContainerAllocator> const> ConstPtr;

}; // struct trajectory_

typedef ::object_msgs::trajectory_<std::allocator<void> > trajectory;

typedef boost::shared_ptr< ::object_msgs::trajectory > trajectoryPtr;
typedef boost::shared_ptr< ::object_msgs::trajectory const> trajectoryConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::object_msgs::trajectory_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::object_msgs::trajectory_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace object_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'object_msgs': ['/home/kbs/xycar_ws/programmers_sdv/project_ws/src/custom_msgs/object_msgs/msg'], 'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::object_msgs::trajectory_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::object_msgs::trajectory_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_msgs::trajectory_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::object_msgs::trajectory_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_msgs::trajectory_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::object_msgs::trajectory_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::object_msgs::trajectory_<ContainerAllocator> >
{
  static const char* value()
  {
    return "6e678bf801db02c4ea2793e8464735ac";
  }

  static const char* value(const ::object_msgs::trajectory_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x6e678bf801db02c4ULL;
  static const uint64_t static_value2 = 0xea2793e8464735acULL;
};

template<class ContainerAllocator>
struct DataType< ::object_msgs::trajectory_<ContainerAllocator> >
{
  static const char* value()
  {
    return "object_msgs/trajectory";
  }

  static const char* value(const ::object_msgs::trajectory_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::object_msgs::trajectory_<ContainerAllocator> >
{
  static const char* value()
  {
    return "std_msgs/Header header\n\
int8 id                 # car id\n\
bool[] mask             # validity : 1 or 0\n\
int8 valid_sequence_num # ex) valid_sequence_num=4 means that only first 4 sequences are valid\n\
Object[] trajectory     # \n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
\n\
================================================================================\n\
MSG: object_msgs/Object\n\
std_msgs/Header header\n\
uint32 id\n\
\n\
# The type of classification given to this object.\n\
uint8 classification\n\
uint8 CLASSIFICATION_UNKNOWN=0\n\
uint8 CLASSIFICATION_CAR=1\n\
uint8 CLASSIFICATION_PEDESTRIAN=2\n\
uint8 CLASSIFICATION_CYCLIST=3\n\
\n\
# The detected position and orientation of the object.\n\
float32 x       # m\n\
float32 y       # m\n\
float32 yaw     # rad\n\
\n\
float32 v       # m/s\n\
float32 yawrate # rad/s\n\
\n\
float32 a      # m/ss\n\
float32 delta  # radian\n\
\n\
# size\n\
float32 L     # m\n\
float32 W     # m\n\
";
  }

  static const char* value(const ::object_msgs::trajectory_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::object_msgs::trajectory_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.id);
      stream.next(m.mask);
      stream.next(m.valid_sequence_num);
      stream.next(m.trajectory);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct trajectory_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::object_msgs::trajectory_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::object_msgs::trajectory_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "id: ";
    Printer<int8_t>::stream(s, indent + "  ", v.id);
    s << indent << "mask[]" << std::endl;
    for (size_t i = 0; i < v.mask.size(); ++i)
    {
      s << indent << "  mask[" << i << "]: ";
      Printer<uint8_t>::stream(s, indent + "  ", v.mask[i]);
    }
    s << indent << "valid_sequence_num: ";
    Printer<int8_t>::stream(s, indent + "  ", v.valid_sequence_num);
    s << indent << "trajectory[]" << std::endl;
    for (size_t i = 0; i < v.trajectory.size(); ++i)
    {
      s << indent << "  trajectory[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::object_msgs::Object_<ContainerAllocator> >::stream(s, indent + "    ", v.trajectory[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // OBJECT_MSGS_MESSAGE_TRAJECTORY_H