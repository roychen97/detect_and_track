node {
  name: "input_image"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 3
        }
        dim {
          size: 300
        }
        dim {
          size: 300
        }
      }
    }
  }
}
node {
  name: "get_layer_anchors/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors/range/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 19
      }
    }
  }
}
node {
  name: "get_layer_anchors/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors/range"
  op: "Range"
  input: "get_layer_anchors/range/start"
  input: "get_layer_anchors/range/limit"
  input: "get_layer_anchors/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors/range_1/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 19
      }
    }
  }
}
node {
  name: "get_layer_anchors/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors/range_1"
  op: "Range"
  input: "get_layer_anchors/range_1/start"
  input: "get_layer_anchors/range_1/limit"
  input: "get_layer_anchors/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Reshape"
  op: "Reshape"
  input: "get_layer_anchors/range"
  input: "get_layer_anchors/meshgrid/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Reshape_1"
  op: "Reshape"
  input: "get_layer_anchors/range_1"
  input: "get_layer_anchors/meshgrid/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 19
      }
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Size_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 19
      }
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Reshape_2"
  op: "Reshape"
  input: "get_layer_anchors/meshgrid/Reshape"
  input: "get_layer_anchors/meshgrid/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Reshape_3/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/Reshape_3"
  op: "Reshape"
  input: "get_layer_anchors/meshgrid/Reshape_1"
  input: "get_layer_anchors/meshgrid/Reshape_3/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/ones/mul"
  op: "Mul"
  input: "get_layer_anchors/meshgrid/Size_1"
  input: "get_layer_anchors/meshgrid/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/ones/Less/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1000
      }
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/ones/Less"
  op: "Less"
  input: "get_layer_anchors/meshgrid/ones/mul"
  input: "get_layer_anchors/meshgrid/ones/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/ones/packed"
  op: "Pack"
  input: "get_layer_anchors/meshgrid/Size_1"
  input: "get_layer_anchors/meshgrid/Size"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/ones/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/ones"
  op: "Fill"
  input: "get_layer_anchors/meshgrid/ones/packed"
  input: "get_layer_anchors/meshgrid/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/mul"
  op: "Mul"
  input: "get_layer_anchors/meshgrid/Reshape_2"
  input: "get_layer_anchors/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/meshgrid/mul_1"
  op: "Mul"
  input: "get_layer_anchors/meshgrid/Reshape_3"
  input: "get_layer_anchors/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/Cast"
  op: "Cast"
  input: "get_layer_anchors/meshgrid/mul_1"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors/add"
  op: "AddV2"
  input: "get_layer_anchors/Cast"
  input: "get_layer_anchors/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 16.0
      }
    }
  }
}
node {
  name: "get_layer_anchors/mul"
  op: "Mul"
  input: "get_layer_anchors/add"
  input: "get_layer_anchors/mul/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors/truediv/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors/truediv"
  op: "RealDiv"
  input: "get_layer_anchors/mul"
  input: "get_layer_anchors/truediv/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors/Cast_1"
  op: "Cast"
  input: "get_layer_anchors/meshgrid/mul"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors/add_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors/add_1"
  op: "AddV2"
  input: "get_layer_anchors/Cast_1"
  input: "get_layer_anchors/add_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 16.0
      }
    }
  }
}
node {
  name: "get_layer_anchors/mul_1"
  op: "Mul"
  input: "get_layer_anchors/add_1"
  input: "get_layer_anchors/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors/truediv_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors/truediv_1"
  op: "RealDiv"
  input: "get_layer_anchors/mul_1"
  input: "get_layer_anchors/truediv_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors/ExpandDims"
  op: "ExpandDims"
  input: "get_layer_anchors/truediv"
  input: "get_layer_anchors/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/ExpandDims_1/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors/ExpandDims_1"
  op: "ExpandDims"
  input: "get_layer_anchors/truediv_1"
  input: "get_layer_anchors/ExpandDims_1/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: ")\313\020>\315\314\314=\303\320\220=\303\320\020>"
      }
    }
  }
}
node {
  name: "get_layer_anchors/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: ")\313\020>\315\314\314=\303\320\020>\303\320\220="
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/range/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/range"
  op: "Range"
  input: "get_layer_anchors_1/range/start"
  input: "get_layer_anchors_1/range/limit"
  input: "get_layer_anchors_1/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/range_1/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/range_1"
  op: "Range"
  input: "get_layer_anchors_1/range_1/start"
  input: "get_layer_anchors_1/range_1/limit"
  input: "get_layer_anchors_1/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Reshape"
  op: "Reshape"
  input: "get_layer_anchors_1/range"
  input: "get_layer_anchors_1/meshgrid/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Reshape_1"
  op: "Reshape"
  input: "get_layer_anchors_1/range_1"
  input: "get_layer_anchors_1/meshgrid/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Size_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Reshape_2"
  op: "Reshape"
  input: "get_layer_anchors_1/meshgrid/Reshape"
  input: "get_layer_anchors_1/meshgrid/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Reshape_3/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/Reshape_3"
  op: "Reshape"
  input: "get_layer_anchors_1/meshgrid/Reshape_1"
  input: "get_layer_anchors_1/meshgrid/Reshape_3/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/ones/mul"
  op: "Mul"
  input: "get_layer_anchors_1/meshgrid/Size_1"
  input: "get_layer_anchors_1/meshgrid/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/ones/Less/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1000
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/ones/Less"
  op: "Less"
  input: "get_layer_anchors_1/meshgrid/ones/mul"
  input: "get_layer_anchors_1/meshgrid/ones/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/ones/packed"
  op: "Pack"
  input: "get_layer_anchors_1/meshgrid/Size_1"
  input: "get_layer_anchors_1/meshgrid/Size"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/ones/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/ones"
  op: "Fill"
  input: "get_layer_anchors_1/meshgrid/ones/packed"
  input: "get_layer_anchors_1/meshgrid/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/mul"
  op: "Mul"
  input: "get_layer_anchors_1/meshgrid/Reshape_2"
  input: "get_layer_anchors_1/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/meshgrid/mul_1"
  op: "Mul"
  input: "get_layer_anchors_1/meshgrid/Reshape_3"
  input: "get_layer_anchors_1/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/Cast"
  op: "Cast"
  input: "get_layer_anchors_1/meshgrid/mul_1"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_1/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/add"
  op: "AddV2"
  input: "get_layer_anchors_1/Cast"
  input: "get_layer_anchors_1/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_1/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 32.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/mul"
  op: "Mul"
  input: "get_layer_anchors_1/add"
  input: "get_layer_anchors_1/mul/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_1/truediv/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/truediv"
  op: "RealDiv"
  input: "get_layer_anchors_1/mul"
  input: "get_layer_anchors_1/truediv/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_1/Cast_1"
  op: "Cast"
  input: "get_layer_anchors_1/meshgrid/mul"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_1/add_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/add_1"
  op: "AddV2"
  input: "get_layer_anchors_1/Cast_1"
  input: "get_layer_anchors_1/add_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_1/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 32.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/mul_1"
  op: "Mul"
  input: "get_layer_anchors_1/add_1"
  input: "get_layer_anchors_1/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_1/truediv_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/truediv_1"
  op: "RealDiv"
  input: "get_layer_anchors_1/mul_1"
  input: "get_layer_anchors_1/truediv_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_1/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/ExpandDims"
  op: "ExpandDims"
  input: "get_layer_anchors_1/truediv"
  input: "get_layer_anchors_1/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/ExpandDims_1/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/ExpandDims_1"
  op: "ExpandDims"
  input: "get_layer_anchors_1/truediv_1"
  input: "get_layer_anchors_1/ExpandDims_1/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_1/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 6
          }
        }
        tensor_content: "\237<\214>\315\314L>\303\320\020>\220{\354=\303\320\220>\361^\261>"
      }
    }
  }
}
node {
  name: "get_layer_anchors_1/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 6
          }
        }
        tensor_content: "\237<\214>\315\314L>\303\320\220>\254\\\261>\303\320\020>\211x\354="
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/range/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 5
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/range"
  op: "Range"
  input: "get_layer_anchors_2/range/start"
  input: "get_layer_anchors_2/range/limit"
  input: "get_layer_anchors_2/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/range_1/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 5
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/range_1"
  op: "Range"
  input: "get_layer_anchors_2/range_1/start"
  input: "get_layer_anchors_2/range_1/limit"
  input: "get_layer_anchors_2/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Reshape"
  op: "Reshape"
  input: "get_layer_anchors_2/range"
  input: "get_layer_anchors_2/meshgrid/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Reshape_1"
  op: "Reshape"
  input: "get_layer_anchors_2/range_1"
  input: "get_layer_anchors_2/meshgrid/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 5
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Size_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 5
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Reshape_2"
  op: "Reshape"
  input: "get_layer_anchors_2/meshgrid/Reshape"
  input: "get_layer_anchors_2/meshgrid/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Reshape_3/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/Reshape_3"
  op: "Reshape"
  input: "get_layer_anchors_2/meshgrid/Reshape_1"
  input: "get_layer_anchors_2/meshgrid/Reshape_3/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/ones/mul"
  op: "Mul"
  input: "get_layer_anchors_2/meshgrid/Size_1"
  input: "get_layer_anchors_2/meshgrid/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/ones/Less/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1000
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/ones/Less"
  op: "Less"
  input: "get_layer_anchors_2/meshgrid/ones/mul"
  input: "get_layer_anchors_2/meshgrid/ones/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/ones/packed"
  op: "Pack"
  input: "get_layer_anchors_2/meshgrid/Size_1"
  input: "get_layer_anchors_2/meshgrid/Size"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/ones/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/ones"
  op: "Fill"
  input: "get_layer_anchors_2/meshgrid/ones/packed"
  input: "get_layer_anchors_2/meshgrid/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/mul"
  op: "Mul"
  input: "get_layer_anchors_2/meshgrid/Reshape_2"
  input: "get_layer_anchors_2/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/meshgrid/mul_1"
  op: "Mul"
  input: "get_layer_anchors_2/meshgrid/Reshape_3"
  input: "get_layer_anchors_2/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/Cast"
  op: "Cast"
  input: "get_layer_anchors_2/meshgrid/mul_1"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_2/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/add"
  op: "AddV2"
  input: "get_layer_anchors_2/Cast"
  input: "get_layer_anchors_2/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_2/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 64.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/mul"
  op: "Mul"
  input: "get_layer_anchors_2/add"
  input: "get_layer_anchors_2/mul/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_2/truediv/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/truediv"
  op: "RealDiv"
  input: "get_layer_anchors_2/mul"
  input: "get_layer_anchors_2/truediv/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_2/Cast_1"
  op: "Cast"
  input: "get_layer_anchors_2/meshgrid/mul"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_2/add_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/add_1"
  op: "AddV2"
  input: "get_layer_anchors_2/Cast_1"
  input: "get_layer_anchors_2/add_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_2/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 64.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/mul_1"
  op: "Mul"
  input: "get_layer_anchors_2/add_1"
  input: "get_layer_anchors_2/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_2/truediv_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/truediv_1"
  op: "RealDiv"
  input: "get_layer_anchors_2/mul_1"
  input: "get_layer_anchors_2/truediv_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_2/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/ExpandDims"
  op: "ExpandDims"
  input: "get_layer_anchors_2/truediv"
  input: "get_layer_anchors_2/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/ExpandDims_1/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/ExpandDims_1"
  op: "ExpandDims"
  input: "get_layer_anchors_2/truediv_1"
  input: "get_layer_anchors_2/ExpandDims_1/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_2/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 6
          }
        }
        tensor_content: "\314\177\350>\000\000\300>\266\303\207>\327\263]>\266\303\007?\002I&?"
      }
    }
  }
}
node {
  name: "get_layer_anchors_2/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 6
          }
        }
        tensor_content: "\314\177\350>\000\000\300>\266\303\007?\341F&?\266\303\207>\001\261]>"
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/range/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/range"
  op: "Range"
  input: "get_layer_anchors_3/range/start"
  input: "get_layer_anchors_3/range/limit"
  input: "get_layer_anchors_3/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/range_1/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/range_1"
  op: "Range"
  input: "get_layer_anchors_3/range_1/start"
  input: "get_layer_anchors_3/range_1/limit"
  input: "get_layer_anchors_3/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Reshape"
  op: "Reshape"
  input: "get_layer_anchors_3/range"
  input: "get_layer_anchors_3/meshgrid/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Reshape_1"
  op: "Reshape"
  input: "get_layer_anchors_3/range_1"
  input: "get_layer_anchors_3/meshgrid/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Size_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 3
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Reshape_2"
  op: "Reshape"
  input: "get_layer_anchors_3/meshgrid/Reshape"
  input: "get_layer_anchors_3/meshgrid/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Reshape_3/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/Reshape_3"
  op: "Reshape"
  input: "get_layer_anchors_3/meshgrid/Reshape_1"
  input: "get_layer_anchors_3/meshgrid/Reshape_3/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/ones/mul"
  op: "Mul"
  input: "get_layer_anchors_3/meshgrid/Size_1"
  input: "get_layer_anchors_3/meshgrid/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/ones/Less/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1000
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/ones/Less"
  op: "Less"
  input: "get_layer_anchors_3/meshgrid/ones/mul"
  input: "get_layer_anchors_3/meshgrid/ones/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/ones/packed"
  op: "Pack"
  input: "get_layer_anchors_3/meshgrid/Size_1"
  input: "get_layer_anchors_3/meshgrid/Size"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/ones/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/ones"
  op: "Fill"
  input: "get_layer_anchors_3/meshgrid/ones/packed"
  input: "get_layer_anchors_3/meshgrid/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/mul"
  op: "Mul"
  input: "get_layer_anchors_3/meshgrid/Reshape_2"
  input: "get_layer_anchors_3/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/meshgrid/mul_1"
  op: "Mul"
  input: "get_layer_anchors_3/meshgrid/Reshape_3"
  input: "get_layer_anchors_3/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/Cast"
  op: "Cast"
  input: "get_layer_anchors_3/meshgrid/mul_1"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_3/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/add"
  op: "AddV2"
  input: "get_layer_anchors_3/Cast"
  input: "get_layer_anchors_3/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_3/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 100.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/mul"
  op: "Mul"
  input: "get_layer_anchors_3/add"
  input: "get_layer_anchors_3/mul/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_3/truediv/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/truediv"
  op: "RealDiv"
  input: "get_layer_anchors_3/mul"
  input: "get_layer_anchors_3/truediv/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_3/Cast_1"
  op: "Cast"
  input: "get_layer_anchors_3/meshgrid/mul"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_3/add_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/add_1"
  op: "AddV2"
  input: "get_layer_anchors_3/Cast_1"
  input: "get_layer_anchors_3/add_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_3/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 100.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/mul_1"
  op: "Mul"
  input: "get_layer_anchors_3/add_1"
  input: "get_layer_anchors_3/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_3/truediv_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/truediv_1"
  op: "RealDiv"
  input: "get_layer_anchors_3/mul_1"
  input: "get_layer_anchors_3/truediv_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_3/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/ExpandDims"
  op: "ExpandDims"
  input: "get_layer_anchors_3/truediv"
  input: "get_layer_anchors_3/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/ExpandDims_1/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/ExpandDims_1"
  op: "ExpandDims"
  input: "get_layer_anchors_3/truediv_1"
  input: "get_layer_anchors_3/ExpandDims_1/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_3/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 6
          }
        }
        tensor_content: "\374\251!?\315\314\014?\014\037\307>\363\224\242>\014\037G?\214\342s?"
      }
    }
  }
}
node {
  name: "get_layer_anchors_3/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 6
          }
        }
        tensor_content: "\374\251!?\315\314\014?\014\037G?m\337s?\014\037\307>\336\222\242>"
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/range/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/range"
  op: "Range"
  input: "get_layer_anchors_4/range/start"
  input: "get_layer_anchors_4/range/limit"
  input: "get_layer_anchors_4/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/range_1/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/range_1"
  op: "Range"
  input: "get_layer_anchors_4/range_1/start"
  input: "get_layer_anchors_4/range_1/limit"
  input: "get_layer_anchors_4/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Reshape"
  op: "Reshape"
  input: "get_layer_anchors_4/range"
  input: "get_layer_anchors_4/meshgrid/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Reshape_1"
  op: "Reshape"
  input: "get_layer_anchors_4/range_1"
  input: "get_layer_anchors_4/meshgrid/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Size_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 2
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Reshape_2"
  op: "Reshape"
  input: "get_layer_anchors_4/meshgrid/Reshape"
  input: "get_layer_anchors_4/meshgrid/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Reshape_3/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/Reshape_3"
  op: "Reshape"
  input: "get_layer_anchors_4/meshgrid/Reshape_1"
  input: "get_layer_anchors_4/meshgrid/Reshape_3/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/ones/mul"
  op: "Mul"
  input: "get_layer_anchors_4/meshgrid/Size_1"
  input: "get_layer_anchors_4/meshgrid/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/ones/Less/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1000
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/ones/Less"
  op: "Less"
  input: "get_layer_anchors_4/meshgrid/ones/mul"
  input: "get_layer_anchors_4/meshgrid/ones/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/ones/packed"
  op: "Pack"
  input: "get_layer_anchors_4/meshgrid/Size_1"
  input: "get_layer_anchors_4/meshgrid/Size"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/ones/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/ones"
  op: "Fill"
  input: "get_layer_anchors_4/meshgrid/ones/packed"
  input: "get_layer_anchors_4/meshgrid/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/mul"
  op: "Mul"
  input: "get_layer_anchors_4/meshgrid/Reshape_2"
  input: "get_layer_anchors_4/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/meshgrid/mul_1"
  op: "Mul"
  input: "get_layer_anchors_4/meshgrid/Reshape_3"
  input: "get_layer_anchors_4/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/Cast"
  op: "Cast"
  input: "get_layer_anchors_4/meshgrid/mul_1"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_4/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/add"
  op: "AddV2"
  input: "get_layer_anchors_4/Cast"
  input: "get_layer_anchors_4/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_4/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 150.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/mul"
  op: "Mul"
  input: "get_layer_anchors_4/add"
  input: "get_layer_anchors_4/mul/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_4/truediv/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/truediv"
  op: "RealDiv"
  input: "get_layer_anchors_4/mul"
  input: "get_layer_anchors_4/truediv/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_4/Cast_1"
  op: "Cast"
  input: "get_layer_anchors_4/meshgrid/mul"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_4/add_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/add_1"
  op: "AddV2"
  input: "get_layer_anchors_4/Cast_1"
  input: "get_layer_anchors_4/add_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_4/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 150.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/mul_1"
  op: "Mul"
  input: "get_layer_anchors_4/add_1"
  input: "get_layer_anchors_4/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_4/truediv_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/truediv_1"
  op: "RealDiv"
  input: "get_layer_anchors_4/mul_1"
  input: "get_layer_anchors_4/truediv_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_4/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/ExpandDims"
  op: "ExpandDims"
  input: "get_layer_anchors_4/truediv"
  input: "get_layer_anchors_4/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/ExpandDims_1/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/ExpandDims_1"
  op: "ExpandDims"
  input: "get_layer_anchors_4/truediv_1"
  input: "get_layer_anchors_4/ExpandDims_1/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_4/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\373\313N?\232\2319?0=\003?0=\203?"
      }
    }
  }
}
node {
  name: "get_layer_anchors_4/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\373\313N?\232\2319?0=\203?0=\003?"
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/range/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/range/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/range/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/range"
  op: "Range"
  input: "get_layer_anchors_5/range/start"
  input: "get_layer_anchors_5/range/limit"
  input: "get_layer_anchors_5/range/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/range_1/start"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/range_1/limit"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/range_1/delta"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/range_1"
  op: "Range"
  input: "get_layer_anchors_5/range_1/start"
  input: "get_layer_anchors_5/range_1/limit"
  input: "get_layer_anchors_5/range_1/delta"
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Reshape"
  op: "Reshape"
  input: "get_layer_anchors_5/range"
  input: "get_layer_anchors_5/meshgrid/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Reshape_1"
  op: "Reshape"
  input: "get_layer_anchors_5/range_1"
  input: "get_layer_anchors_5/meshgrid/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Size"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Size_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Reshape_2"
  op: "Reshape"
  input: "get_layer_anchors_5/meshgrid/Reshape"
  input: "get_layer_anchors_5/meshgrid/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Reshape_3/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\377\377\377\377\001\000\000\000"
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/Reshape_3"
  op: "Reshape"
  input: "get_layer_anchors_5/meshgrid/Reshape_1"
  input: "get_layer_anchors_5/meshgrid/Reshape_3/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/ones/mul"
  op: "Mul"
  input: "get_layer_anchors_5/meshgrid/Size_1"
  input: "get_layer_anchors_5/meshgrid/Size"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/ones/Less/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1000
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/ones/Less"
  op: "Less"
  input: "get_layer_anchors_5/meshgrid/ones/mul"
  input: "get_layer_anchors_5/meshgrid/ones/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/ones/packed"
  op: "Pack"
  input: "get_layer_anchors_5/meshgrid/Size_1"
  input: "get_layer_anchors_5/meshgrid/Size"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/ones/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/ones"
  op: "Fill"
  input: "get_layer_anchors_5/meshgrid/ones/packed"
  input: "get_layer_anchors_5/meshgrid/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/mul"
  op: "Mul"
  input: "get_layer_anchors_5/meshgrid/Reshape_2"
  input: "get_layer_anchors_5/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/meshgrid/mul_1"
  op: "Mul"
  input: "get_layer_anchors_5/meshgrid/Reshape_3"
  input: "get_layer_anchors_5/meshgrid/ones"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/Cast"
  op: "Cast"
  input: "get_layer_anchors_5/meshgrid/mul_1"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_5/add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/add"
  op: "AddV2"
  input: "get_layer_anchors_5/Cast"
  input: "get_layer_anchors_5/add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_5/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/mul"
  op: "Mul"
  input: "get_layer_anchors_5/add"
  input: "get_layer_anchors_5/mul/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_5/truediv/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/truediv"
  op: "RealDiv"
  input: "get_layer_anchors_5/mul"
  input: "get_layer_anchors_5/truediv/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_5/Cast_1"
  op: "Cast"
  input: "get_layer_anchors_5/meshgrid/mul"
  attr {
    key: "DstT"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "SrcT"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Truncate"
    value {
      b: false
    }
  }
}
node {
  name: "get_layer_anchors_5/add_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.5
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/add_1"
  op: "AddV2"
  input: "get_layer_anchors_5/Cast_1"
  input: "get_layer_anchors_5/add_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_5/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/mul_1"
  op: "Mul"
  input: "get_layer_anchors_5/add_1"
  input: "get_layer_anchors_5/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_5/truediv_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 300.0
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/truediv_1"
  op: "RealDiv"
  input: "get_layer_anchors_5/mul_1"
  input: "get_layer_anchors_5/truediv_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "get_layer_anchors_5/ExpandDims/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/ExpandDims"
  op: "ExpandDims"
  input: "get_layer_anchors_5/truediv"
  input: "get_layer_anchors_5/ExpandDims/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/ExpandDims_1/dim"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/ExpandDims_1"
  op: "ExpandDims"
  input: "get_layer_anchors_5/truediv_1"
  input: "get_layer_anchors_5/ExpandDims_1/dim"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tdim"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "get_layer_anchors_5/Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "6\315{?fff?\333\352\"?\333\352\242?"
      }
    }
  }
}
node {
  name: "get_layer_anchors_5/Const_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "6\315{?fff?\333\352\242?\333\352\"?"
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv"
  op: "RealDiv"
  input: "get_layer_anchors/Const"
  input: "ext_decode_all_anchors/truediv/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub"
  op: "Sub"
  input: "get_layer_anchors/ExpandDims"
  input: "ext_decode_all_anchors/truediv"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_1"
  op: "RealDiv"
  input: "get_layer_anchors/Const_1"
  input: "ext_decode_all_anchors/truediv_1/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_1"
  op: "Sub"
  input: "get_layer_anchors/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_2/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_2"
  op: "RealDiv"
  input: "get_layer_anchors/Const"
  input: "ext_decode_all_anchors/truediv_2/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add"
  op: "AddV2"
  input: "get_layer_anchors/ExpandDims"
  input: "ext_decode_all_anchors/truediv_2"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_3/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_3"
  op: "RealDiv"
  input: "get_layer_anchors/Const_1"
  input: "ext_decode_all_anchors/truediv_3/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_1"
  op: "AddV2"
  input: "get_layer_anchors/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_3"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub"
  input: "ext_decode_all_anchors/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_1"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_1"
  input: "ext_decode_all_anchors/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_2"
  op: "Reshape"
  input: "ext_decode_all_anchors/add"
  input: "ext_decode_all_anchors/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_3/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_3"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_1"
  input: "ext_decode_all_anchors/Reshape_3/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_4/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_4"
  op: "RealDiv"
  input: "get_layer_anchors_1/Const"
  input: "ext_decode_all_anchors/truediv_4/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_2"
  op: "Sub"
  input: "get_layer_anchors_1/ExpandDims"
  input: "ext_decode_all_anchors/truediv_4"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_5/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_5"
  op: "RealDiv"
  input: "get_layer_anchors_1/Const_1"
  input: "ext_decode_all_anchors/truediv_5/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_3"
  op: "Sub"
  input: "get_layer_anchors_1/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_5"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_6/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_6"
  op: "RealDiv"
  input: "get_layer_anchors_1/Const"
  input: "ext_decode_all_anchors/truediv_6/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_2"
  op: "AddV2"
  input: "get_layer_anchors_1/ExpandDims"
  input: "ext_decode_all_anchors/truediv_6"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_7/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_7"
  op: "RealDiv"
  input: "get_layer_anchors_1/Const_1"
  input: "ext_decode_all_anchors/truediv_7/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_3"
  op: "AddV2"
  input: "get_layer_anchors_1/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_7"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_4/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_4"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_2"
  input: "ext_decode_all_anchors/Reshape_4/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_5/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_5"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_3"
  input: "ext_decode_all_anchors/Reshape_5/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_6/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_6"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_2"
  input: "ext_decode_all_anchors/Reshape_6/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_7/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_7"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_3"
  input: "ext_decode_all_anchors/Reshape_7/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_8/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_8"
  op: "RealDiv"
  input: "get_layer_anchors_2/Const"
  input: "ext_decode_all_anchors/truediv_8/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_4"
  op: "Sub"
  input: "get_layer_anchors_2/ExpandDims"
  input: "ext_decode_all_anchors/truediv_8"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_9/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_9"
  op: "RealDiv"
  input: "get_layer_anchors_2/Const_1"
  input: "ext_decode_all_anchors/truediv_9/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_5"
  op: "Sub"
  input: "get_layer_anchors_2/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_9"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_10/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_10"
  op: "RealDiv"
  input: "get_layer_anchors_2/Const"
  input: "ext_decode_all_anchors/truediv_10/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_4"
  op: "AddV2"
  input: "get_layer_anchors_2/ExpandDims"
  input: "ext_decode_all_anchors/truediv_10"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_11/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_11"
  op: "RealDiv"
  input: "get_layer_anchors_2/Const_1"
  input: "ext_decode_all_anchors/truediv_11/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_5"
  op: "AddV2"
  input: "get_layer_anchors_2/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_11"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_8/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_8"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_4"
  input: "ext_decode_all_anchors/Reshape_8/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_9/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_9"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_5"
  input: "ext_decode_all_anchors/Reshape_9/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_10/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_10"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_4"
  input: "ext_decode_all_anchors/Reshape_10/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_11/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_11"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_5"
  input: "ext_decode_all_anchors/Reshape_11/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_12/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_12"
  op: "RealDiv"
  input: "get_layer_anchors_3/Const"
  input: "ext_decode_all_anchors/truediv_12/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_6"
  op: "Sub"
  input: "get_layer_anchors_3/ExpandDims"
  input: "ext_decode_all_anchors/truediv_12"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_13/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_13"
  op: "RealDiv"
  input: "get_layer_anchors_3/Const_1"
  input: "ext_decode_all_anchors/truediv_13/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_7"
  op: "Sub"
  input: "get_layer_anchors_3/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_13"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_14/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_14"
  op: "RealDiv"
  input: "get_layer_anchors_3/Const"
  input: "ext_decode_all_anchors/truediv_14/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_6"
  op: "AddV2"
  input: "get_layer_anchors_3/ExpandDims"
  input: "ext_decode_all_anchors/truediv_14"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_15/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_15"
  op: "RealDiv"
  input: "get_layer_anchors_3/Const_1"
  input: "ext_decode_all_anchors/truediv_15/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_7"
  op: "AddV2"
  input: "get_layer_anchors_3/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_15"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_12/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_12"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_6"
  input: "ext_decode_all_anchors/Reshape_12/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_13/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_13"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_7"
  input: "ext_decode_all_anchors/Reshape_13/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_14/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_14"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_6"
  input: "ext_decode_all_anchors/Reshape_14/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_15/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_15"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_7"
  input: "ext_decode_all_anchors/Reshape_15/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_16/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_16"
  op: "RealDiv"
  input: "get_layer_anchors_4/Const"
  input: "ext_decode_all_anchors/truediv_16/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_8"
  op: "Sub"
  input: "get_layer_anchors_4/ExpandDims"
  input: "ext_decode_all_anchors/truediv_16"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_17/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_17"
  op: "RealDiv"
  input: "get_layer_anchors_4/Const_1"
  input: "ext_decode_all_anchors/truediv_17/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_9"
  op: "Sub"
  input: "get_layer_anchors_4/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_17"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_18/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_18"
  op: "RealDiv"
  input: "get_layer_anchors_4/Const"
  input: "ext_decode_all_anchors/truediv_18/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_8"
  op: "AddV2"
  input: "get_layer_anchors_4/ExpandDims"
  input: "ext_decode_all_anchors/truediv_18"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_19/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_19"
  op: "RealDiv"
  input: "get_layer_anchors_4/Const_1"
  input: "ext_decode_all_anchors/truediv_19/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_9"
  op: "AddV2"
  input: "get_layer_anchors_4/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_19"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_16/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_16"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_8"
  input: "ext_decode_all_anchors/Reshape_16/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_17/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_17"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_9"
  input: "ext_decode_all_anchors/Reshape_17/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_18/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_18"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_8"
  input: "ext_decode_all_anchors/Reshape_18/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_19/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_19"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_9"
  input: "ext_decode_all_anchors/Reshape_19/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_20/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_20"
  op: "RealDiv"
  input: "get_layer_anchors_5/Const"
  input: "ext_decode_all_anchors/truediv_20/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_10"
  op: "Sub"
  input: "get_layer_anchors_5/ExpandDims"
  input: "ext_decode_all_anchors/truediv_20"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_21/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_21"
  op: "RealDiv"
  input: "get_layer_anchors_5/Const_1"
  input: "ext_decode_all_anchors/truediv_21/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/sub_11"
  op: "Sub"
  input: "get_layer_anchors_5/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_21"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_22/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_22"
  op: "RealDiv"
  input: "get_layer_anchors_5/Const"
  input: "ext_decode_all_anchors/truediv_22/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_10"
  op: "AddV2"
  input: "get_layer_anchors_5/ExpandDims"
  input: "ext_decode_all_anchors/truediv_22"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_23/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 2.0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/truediv_23"
  op: "RealDiv"
  input: "get_layer_anchors_5/Const_1"
  input: "ext_decode_all_anchors/truediv_23/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/add_11"
  op: "AddV2"
  input: "get_layer_anchors_5/ExpandDims_1"
  input: "ext_decode_all_anchors/truediv_23"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_20/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_20"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_10"
  input: "ext_decode_all_anchors/Reshape_20/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_21/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_21"
  op: "Reshape"
  input: "ext_decode_all_anchors/sub_11"
  input: "ext_decode_all_anchors/Reshape_21/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_22/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_22"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_10"
  input: "ext_decode_all_anchors/Reshape_22/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_23/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_23"
  op: "Reshape"
  input: "ext_decode_all_anchors/add_11"
  input: "ext_decode_all_anchors/Reshape_23/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat_ymin/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat_ymin"
  op: "ConcatV2"
  input: "ext_decode_all_anchors/Reshape"
  input: "ext_decode_all_anchors/Reshape_4"
  input: "ext_decode_all_anchors/Reshape_8"
  input: "ext_decode_all_anchors/Reshape_12"
  input: "ext_decode_all_anchors/Reshape_16"
  input: "ext_decode_all_anchors/Reshape_20"
  input: "ext_decode_all_anchors/concat_ymin/axis"
  attr {
    key: "N"
    value {
      i: 6
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat_xmin/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat_xmin"
  op: "ConcatV2"
  input: "ext_decode_all_anchors/Reshape_1"
  input: "ext_decode_all_anchors/Reshape_5"
  input: "ext_decode_all_anchors/Reshape_9"
  input: "ext_decode_all_anchors/Reshape_13"
  input: "ext_decode_all_anchors/Reshape_17"
  input: "ext_decode_all_anchors/Reshape_21"
  input: "ext_decode_all_anchors/concat_xmin/axis"
  attr {
    key: "N"
    value {
      i: 6
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat_ymax/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat_ymax"
  op: "ConcatV2"
  input: "ext_decode_all_anchors/Reshape_2"
  input: "ext_decode_all_anchors/Reshape_6"
  input: "ext_decode_all_anchors/Reshape_10"
  input: "ext_decode_all_anchors/Reshape_14"
  input: "ext_decode_all_anchors/Reshape_18"
  input: "ext_decode_all_anchors/Reshape_22"
  input: "ext_decode_all_anchors/concat_ymax/axis"
  attr {
    key: "N"
    value {
      i: 6
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat_xmax/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat_xmax"
  op: "ConcatV2"
  input: "ext_decode_all_anchors/Reshape_3"
  input: "ext_decode_all_anchors/Reshape_7"
  input: "ext_decode_all_anchors/Reshape_11"
  input: "ext_decode_all_anchors/Reshape_15"
  input: "ext_decode_all_anchors/Reshape_19"
  input: "ext_decode_all_anchors/Reshape_23"
  input: "ext_decode_all_anchors/concat_xmax/axis"
  attr {
    key: "N"
    value {
      i: 6
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 2268
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/strided_slice/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/strided_slice/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/strided_slice/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/strided_slice"
  op: "StridedSlice"
  input: "ext_decode_all_anchors/Shape"
  input: "ext_decode_all_anchors/strided_slice/stack"
  input: "ext_decode_all_anchors/strided_slice/stack_1"
  input: "ext_decode_all_anchors/strided_slice/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 1
    }
  }
}
node {
  name: "ext_decode_all_anchors/Tile/input"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 4
          }
        }
        tensor_content: "\315\314\314=\315\314\314=\315\314L>\315\314L>"
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Tile/multiples/1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Tile/multiples"
  op: "Pack"
  input: "ext_decode_all_anchors/strided_slice"
  input: "ext_decode_all_anchors/Tile/multiples/1"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "ext_decode_all_anchors/Tile"
  op: "Tile"
  input: "ext_decode_all_anchors/Tile/input"
  input: "ext_decode_all_anchors/Tile/multiples"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tmultiples"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/concat"
  op: "ConcatV2"
  input: "ext_decode_all_anchors/concat_xmin"
  input: "ext_decode_all_anchors/concat_ymin"
  input: "ext_decode_all_anchors/concat_xmax"
  input: "ext_decode_all_anchors/concat_ymax"
  input: "ext_decode_all_anchors/concat/axis"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_24/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\004\000\000\000\377\377\377\377"
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_24"
  op: "Reshape"
  input: "ext_decode_all_anchors/concat"
  input: "ext_decode_all_anchors/Reshape_24/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/transpose/perm"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\000\000\000\000"
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/transpose"
  op: "Transpose"
  input: "ext_decode_all_anchors/Reshape_24"
  input: "ext_decode_all_anchors/transpose/perm"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tperm"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/mul/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 4
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/mul"
  op: "Mul"
  input: "ext_decode_all_anchors/strided_slice"
  input: "ext_decode_all_anchors/mul/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_25/shape/0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_25/shape/1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_25/shape/3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_25/shape"
  op: "Pack"
  input: "ext_decode_all_anchors/Reshape_25/shape/0"
  input: "ext_decode_all_anchors/Reshape_25/shape/1"
  input: "ext_decode_all_anchors/mul"
  input: "ext_decode_all_anchors/Reshape_25/shape/3"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_25"
  op: "Reshape"
  input: "ext_decode_all_anchors/transpose"
  input: "ext_decode_all_anchors/Reshape_25/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/mul_1/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 4
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/mul_1"
  op: "Mul"
  input: "ext_decode_all_anchors/strided_slice"
  input: "ext_decode_all_anchors/mul_1/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_26/shape/0"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: -1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_26/shape/1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_26/shape/3"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_26/shape"
  op: "Pack"
  input: "ext_decode_all_anchors/Reshape_26/shape/0"
  input: "ext_decode_all_anchors/Reshape_26/shape/1"
  input: "ext_decode_all_anchors/mul_1"
  input: "ext_decode_all_anchors/Reshape_26/shape/3"
  attr {
    key: "N"
    value {
      i: 4
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "axis"
    value {
      i: 0
    }
  }
}
node {
  name: "ext_decode_all_anchors/Reshape_26"
  op: "Reshape"
  input: "ext_decode_all_anchors/Tile"
  input: "ext_decode_all_anchors/Reshape_26/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/name/axis"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/name"
  op: "ConcatV2"
  input: "ext_decode_all_anchors/Reshape_25"
  input: "ext_decode_all_anchors/Reshape_26"
  input: "ext_decode_all_anchors/name/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ext_decode_all_anchors/strided_slice_1/stack"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/strided_slice_1/stack_1"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000\000\000\000\000\000\000\000\000"
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/strided_slice_1/stack_2"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ext_decode_all_anchors/strided_slice_1"
  op: "StridedSlice"
  input: "ext_decode_all_anchors/name"
  input: "ext_decode_all_anchors/strided_slice_1/stack"
  input: "ext_decode_all_anchors/strided_slice_1/stack_1"
  input: "ext_decode_all_anchors/strided_slice_1/stack_2"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 13
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 13
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 2
    }
  }
}
node {
  name: "ssd300/Conv/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\003\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/Conv/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1380131095647812
      }
    }
  }
}
node {
  name: "ssd300/Conv/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1380131095647812
      }
    }
  }
}
node {
  name: "ssd300/Conv/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/Conv/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/Conv/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/Conv/weights/Initializer/random_uniform/max"
  input: "ssd300/Conv/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
}
node {
  name: "ssd300/Conv/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/Conv/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/Conv/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
}
node {
  name: "ssd300/Conv/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/Conv/weights/Initializer/random_uniform/mul"
  input: "ssd300/Conv/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
}
node {
  name: "ssd300/Conv/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/Conv/weights/Assign"
  op: "Assign"
  input: "ssd300/Conv/weights"
  input: "ssd300/Conv/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv/weights/read"
  op: "Identity"
  input: "ssd300/Conv/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/weights"
      }
    }
  }
}
node {
  name: "ssd300/Conv/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "input_image"
  input: "ssd300/Conv/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 2
        i: 2
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/Conv/BatchNorm/beta"
  input: "ssd300/Conv/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/Conv/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/Conv/BatchNorm/moving_mean"
  input: "ssd300/Conv/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/Conv/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/Conv/BatchNorm/moving_variance"
  input: "ssd300/Conv/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/Conv/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/Conv/BatchNorm/beta/read"
  input: "ssd300/Conv/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/Conv/BatchNorm/moving_mean/read"
  input: "ssd300/Conv/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/Conv/BatchNorm/moving_variance/read"
  input: "ssd300/Conv/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/Conv/BatchNorm/Reshape_2"
  input: "ssd300/Conv/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/Conv/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/Conv/layer_conv2d/Conv2D"
  input: "ssd300/Conv/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/Conv/BatchNorm/Reshape_1"
  input: "ssd300/Conv/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/Conv/BatchNorm/Reshape"
  input: "ssd300/Conv/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/Conv/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/Conv/BatchNorm/batchnorm_1/mul"
  input: "ssd300/Conv/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/Conv/Relu6"
  op: "Relu6"
  input: "ssd300/Conv/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.14213381707668304
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.14213381707668304
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 32
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000 \000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/Conv/Relu6"
  input: "ssd300/expanded_conv/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000 \000\000\000\020\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.3535533845424652
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.3535533845424652
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 32
        }
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv/project/weights"
  input: "ssd300/expanded_conv/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv/depthwise/Relu6"
  input: "ssd300/expanded_conv/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv/project/BatchNorm/beta"
  input: "ssd300/expanded_conv/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 16
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 16
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\020\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\020\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\020\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\020\000\000\000`\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.2314550280570984
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.2314550280570984
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 16
        }
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/expand/weights"
  input: "ssd300/expanded_conv_1/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_1/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_1/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_1/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000`\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.08290266990661621
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.08290266990661621
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 96
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000`\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_1/expand/Relu6"
  input: "ssd300/expanded_conv_1/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 2
        i: 2
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_1/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_1/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000`\000\000\000\030\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.22360679507255554
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.22360679507255554
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 96
        }
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/project/weights"
  input: "ssd300/expanded_conv_1/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_1/depthwise/Relu6"
  input: "ssd300/expanded_conv_1/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 24
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_1/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 24
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_1/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 24
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_1/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_1/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_1/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_1/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\030\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_1/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_1/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\030\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_1/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_1/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\030\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_1/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_1/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_1/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_1/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_1/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_1/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\030\000\000\000\220\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.18898223340511322
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.18898223340511322
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 24
        }
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/expand/weights"
  input: "ssd300/expanded_conv_2/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_2/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_2/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_2/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\220\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.06780634820461273
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.06780634820461273
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 144
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\220\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_2/expand/Relu6"
  input: "ssd300/expanded_conv_2/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_2/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_2/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\220\000\000\000\030\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.18898223340511322
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.18898223340511322
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 144
        }
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/project/weights"
  input: "ssd300/expanded_conv_2/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_2/depthwise/Relu6"
  input: "ssd300/expanded_conv_2/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 24
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_2/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 24
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_2/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 24
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 24
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_2/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_2/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_2/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_2/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\030\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_2/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_2/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\030\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_2/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_2/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\030\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_2/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_2/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_2/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_2/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_2/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_2/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_2/Add"
  op: "Add"
  input: "ssd300/expanded_conv_2/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_1/project/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\030\000\000\000\220\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.18898223340511322
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.18898223340511322
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 24
        }
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/expand/weights"
  input: "ssd300/expanded_conv_3/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_2/Add"
  input: "ssd300/expanded_conv_3/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_3/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_3/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\220\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.06780634820461273
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.06780634820461273
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 144
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\220\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_3/expand/Relu6"
  input: "ssd300/expanded_conv_3/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 2
        i: 2
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 144
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 144
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\220\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_3/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_3/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\220\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.18463723361492157
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.18463723361492157
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 144
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/project/weights"
  input: "ssd300/expanded_conv_3/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_3/depthwise/Relu6"
  input: "ssd300/expanded_conv_3/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_3/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_3/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_3/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_3/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_3/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_3/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_3/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_3/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_3/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_3/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_3/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_3/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_3/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_3/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_3/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_3/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000 \000\000\000\300\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 32
        }
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/expand/weights"
  input: "ssd300/expanded_conv_4/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_4/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_4/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_4/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.05877270922064781
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.05877270922064781
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 192
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_4/expand/Relu6"
  input: "ssd300/expanded_conv_4/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_4/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_4/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\300\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 192
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/project/weights"
  input: "ssd300/expanded_conv_4/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_4/depthwise/Relu6"
  input: "ssd300/expanded_conv_4/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_4/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_4/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_4/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_4/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_4/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_4/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_4/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_4/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_4/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_4/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_4/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_4/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_4/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_4/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_4/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_4/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_4/Add"
  op: "Add"
  input: "ssd300/expanded_conv_4/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_3/project/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000 \000\000\000\300\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 32
        }
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/expand/weights"
  input: "ssd300/expanded_conv_5/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_4/Add"
  input: "ssd300/expanded_conv_5/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_5/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_5/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.05877270922064781
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.05877270922064781
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 192
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_5/expand/Relu6"
  input: "ssd300/expanded_conv_5/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_5/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_5/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\300\000\000\000 \000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 192
        }
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/project/weights"
  input: "ssd300/expanded_conv_5/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_5/depthwise/Relu6"
  input: "ssd300/expanded_conv_5/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_5/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_5/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 32
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_5/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_5/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_5/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_5/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_5/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_5/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_5/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_5/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000 \000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_5/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_5/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_5/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_5/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_5/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_5/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_5/Add"
  op: "Add"
  input: "ssd300/expanded_conv_5/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_4/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000 \000\000\000\300\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.16366341710090637
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 32
        }
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/expand/weights"
  input: "ssd300/expanded_conv_6/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_5/Add"
  input: "ssd300/expanded_conv_6/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_6/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_6/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.05877270922064781
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.05877270922064781
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 192
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_6/expand/Relu6"
  input: "ssd300/expanded_conv_6/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 2
        i: 2
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 192
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 192
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_6/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_6/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\300\000\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1530931144952774
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1530931144952774
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 192
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/project/weights"
  input: "ssd300/expanded_conv_6/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_6/depthwise/Relu6"
  input: "ssd300/expanded_conv_6/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_6/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_6/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_6/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_6/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_6/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_6/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_6/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_6/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_6/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_6/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_6/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_6/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_6/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_6/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_6/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_6/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000@\000\000\000\200\001\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 64
        }
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/expand/weights"
  input: "ssd300/expanded_conv_7/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_7/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_7/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_7/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\001\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.04161251708865166
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.04161251708865166
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 384
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\001\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_7/expand/Relu6"
  input: "ssd300/expanded_conv_7/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_7/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_7/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\001\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 384
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/project/weights"
  input: "ssd300/expanded_conv_7/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_7/depthwise/Relu6"
  input: "ssd300/expanded_conv_7/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_7/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_7/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_7/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_7/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_7/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_7/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_7/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_7/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_7/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_7/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_7/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_7/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_7/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_7/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_7/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_7/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_7/Add"
  op: "Add"
  input: "ssd300/expanded_conv_7/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_6/project/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000@\000\000\000\200\001\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 64
        }
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/expand/weights"
  input: "ssd300/expanded_conv_8/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_7/Add"
  input: "ssd300/expanded_conv_8/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_8/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_8/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\001\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.04161251708865166
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.04161251708865166
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 384
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\001\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_8/expand/Relu6"
  input: "ssd300/expanded_conv_8/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_8/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_8/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\001\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 384
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/project/weights"
  input: "ssd300/expanded_conv_8/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_8/depthwise/Relu6"
  input: "ssd300/expanded_conv_8/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_8/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_8/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_8/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_8/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_8/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_8/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_8/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_8/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_8/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_8/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_8/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_8/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_8/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_8/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_8/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_8/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_8/Add"
  op: "Add"
  input: "ssd300/expanded_conv_8/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_7/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000@\000\000\000\200\001\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 64
        }
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/expand/weights"
  input: "ssd300/expanded_conv_9/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_8/Add"
  input: "ssd300/expanded_conv_9/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_9/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_9/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\001\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.04161251708865166
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.04161251708865166
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 384
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\001\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_9/expand/Relu6"
  input: "ssd300/expanded_conv_9/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_9/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_9/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\001\000\000@\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 384
        }
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/project/weights"
  input: "ssd300/expanded_conv_9/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_9/depthwise/Relu6"
  input: "ssd300/expanded_conv_9/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_9/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_9/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 64
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 64
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_9/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_9/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_9/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_9/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_9/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_9/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_9/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_9/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_9/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_9/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_9/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_9/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_9/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_9/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_9/Add"
  op: "Add"
  input: "ssd300/expanded_conv_9/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_8/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000@\000\000\000\200\001\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.1157275140285492
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 64
        }
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/expand/weights"
  input: "ssd300/expanded_conv_10/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_9/Add"
  input: "ssd300/expanded_conv_10/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_10/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_10/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\001\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.04161251708865166
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.04161251708865166
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 384
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\200\001\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_10/expand/Relu6"
  input: "ssd300/expanded_conv_10/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 384
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 384
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\200\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_10/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_10/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\200\001\000\000`\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.11180339753627777
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.11180339753627777
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 384
        }
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/project/weights"
  input: "ssd300/expanded_conv_10/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_10/depthwise/Relu6"
  input: "ssd300/expanded_conv_10/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_10/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_10/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_10/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_10/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_10/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_10/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_10/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_10/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_10/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_10/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_10/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_10/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_10/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_10/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_10/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_10/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000`\000\000\000@\002\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 96
        }
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/expand/weights"
  input: "ssd300/expanded_conv_11/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_11/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_11/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_11/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\002\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.03399119898676872
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.03399119898676872
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 576
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\002\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_11/expand/Relu6"
  input: "ssd300/expanded_conv_11/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_11/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_11/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000@\002\000\000`\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 576
        }
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/project/weights"
  input: "ssd300/expanded_conv_11/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_11/depthwise/Relu6"
  input: "ssd300/expanded_conv_11/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_11/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_11/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_11/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_11/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_11/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_11/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_11/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_11/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_11/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_11/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_11/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_11/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_11/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_11/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_11/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_11/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_11/Add"
  op: "Add"
  input: "ssd300/expanded_conv_11/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_10/project/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000`\000\000\000@\002\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 96
        }
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/expand/weights"
  input: "ssd300/expanded_conv_12/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_11/Add"
  input: "ssd300/expanded_conv_12/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_12/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_12/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\002\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.03399119898676872
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.03399119898676872
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 576
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\002\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_12/expand/Relu6"
  input: "ssd300/expanded_conv_12/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_12/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_12/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000@\002\000\000`\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 576
        }
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/project/weights"
  input: "ssd300/expanded_conv_12/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_12/depthwise/Relu6"
  input: "ssd300/expanded_conv_12/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_12/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_12/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 96
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 96
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_12/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_12/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_12/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_12/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_12/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_12/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_12/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_12/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000`\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_12/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_12/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_12/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_12/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_12/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_12/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_12/Add"
  op: "Add"
  input: "ssd300/expanded_conv_12/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_11/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000`\000\000\000@\002\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.09449111670255661
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 96
        }
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/expand/weights"
  input: "ssd300/expanded_conv_13/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_12/Add"
  input: "ssd300/expanded_conv_13/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_13/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_13/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\002\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.03399119898676872
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.03399119898676872
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 576
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000@\002\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_13/expand/Relu6"
  input: "ssd300/expanded_conv_13/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 2
        i: 2
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 576
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 576
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\002\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_13/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_13/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000@\002\000\000\240\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.0902893915772438
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0902893915772438
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 576
        }
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/project/weights"
  input: "ssd300/expanded_conv_13/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_13/depthwise/Relu6"
  input: "ssd300/expanded_conv_13/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 160
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_13/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 160
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_13/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 160
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_13/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_13/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_13/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_13/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\240\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_13/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_13/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\240\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_13/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_13/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\240\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_13/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_13/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_13/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_13/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_13/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_13/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\240\000\000\000\300\003\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 160
        }
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/expand/weights"
  input: "ssd300/expanded_conv_14/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_14/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_14/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_14/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\003\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.026338599622249603
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.026338599622249603
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 960
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\003\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_14/expand/Relu6"
  input: "ssd300/expanded_conv_14/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_14/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_14/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\300\003\000\000\240\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 960
        }
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/project/weights"
  input: "ssd300/expanded_conv_14/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_14/depthwise/Relu6"
  input: "ssd300/expanded_conv_14/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 160
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_14/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 160
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_14/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 160
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_14/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_14/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_14/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_14/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\240\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_14/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_14/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\240\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_14/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_14/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\240\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_14/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_14/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_14/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_14/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_14/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_14/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_14/Add"
  op: "Add"
  input: "ssd300/expanded_conv_14/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_13/project/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\240\000\000\000\300\003\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 160
        }
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/expand/weights"
  input: "ssd300/expanded_conv_15/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_14/Add"
  input: "ssd300/expanded_conv_15/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_15/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_15/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\003\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.026338599622249603
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.026338599622249603
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 960
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\003\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_15/expand/Relu6"
  input: "ssd300/expanded_conv_15/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_15/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_15/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\300\003\000\000\240\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 960
        }
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/project/weights"
  input: "ssd300/expanded_conv_15/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_15/depthwise/Relu6"
  input: "ssd300/expanded_conv_15/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 160
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_15/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 160
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_15/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 160
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 160
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_15/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_15/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_15/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_15/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\240\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_15/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_15/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\240\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_15/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_15/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\240\000\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_15/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_15/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_15/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_15/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_15/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_15/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_15/Add"
  op: "Add"
  input: "ssd300/expanded_conv_15/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/expanded_conv_14/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\240\000\000\000\300\003\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.07319250702857971
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 160
        }
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/expand/weights"
  input: "ssd300/expanded_conv_16/expand/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/expand/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_15/Add"
  input: "ssd300/expanded_conv_16/expand/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/beta"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/expand/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_16/expand/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/expand/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_16/expand/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\003\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.026338599622249603
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.026338599622249603
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 960
        }
        dim {
          size: 1
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/depthwise_weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/depthwise_weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/layer_conv2d/depthwise/Shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\003\000\000\000\003\000\000\000\300\003\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/layer_conv2d/depthwise/dilation_rate"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/layer_conv2d/depthwise"
  op: "DepthwiseConv2dNative"
  input: "ssd300/expanded_conv_16/expand/Relu6"
  input: "ssd300/expanded_conv_16/depthwise/depthwise_weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/beta"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 960
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 960
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\300\003\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_16/depthwise/layer_conv2d/depthwise"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/depthwise/Relu6"
  op: "Relu6"
  input: "ssd300/expanded_conv_16/depthwise/BatchNorm/batchnorm_1/Add_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000\300\003\000\000@\001\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.06846532225608826
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.06846532225608826
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/max"
  input: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/mul"
  input: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 960
        }
        dim {
          size: 320
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/project/weights"
  input: "ssd300/expanded_conv_16/project/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/weights/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/project/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/weights"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_16/depthwise/Relu6"
  input: "ssd300/expanded_conv_16/project/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/beta/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 320
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 320
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/project/BatchNorm/beta"
  input: "ssd300/expanded_conv_16/project/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/project/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/moving_mean/Initializer/zeros"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 320
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 320
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/project/BatchNorm/moving_mean"
  input: "ssd300/expanded_conv_16/project/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/project/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/moving_variance/Initializer/ones"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 320
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 320
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/expanded_conv_16/project/BatchNorm/moving_variance"
  input: "ssd300/expanded_conv_16/project/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/expanded_conv_16/project/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/expanded_conv_16/project/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/expanded_conv_16/project/BatchNorm/beta/read"
  input: "ssd300/expanded_conv_16/project/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/expanded_conv_16/project/BatchNorm/moving_mean/read"
  input: "ssd300/expanded_conv_16/project/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000@\001\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/expanded_conv_16/project/BatchNorm/moving_variance/read"
  input: "ssd300/expanded_conv_16/project/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0010000000474974513
      }
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/Add"
  op: "Add"
  input: "ssd300/expanded_conv_16/project/BatchNorm/Reshape_2"
  input: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/Add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/Rsqrt"
  op: "Rsqrt"
  input: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/Add"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/mul"
  op: "Mul"
  input: "ssd300/expanded_conv_16/project/layer_conv2d/Conv2D"
  input: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/mul_1"
  op: "Mul"
  input: "ssd300/expanded_conv_16/project/BatchNorm/Reshape_1"
  input: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/Rsqrt"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/sub"
  op: "Sub"
  input: "ssd300/expanded_conv_16/project/BatchNorm/Reshape"
  input: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/mul_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/Add_1"
  op: "Add"
  input: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/mul"
  input: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "ssd300/Conv_1/weights/Initializer/random_uniform/shape"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\001\000\000\000@\001\000\000\000\005\000\000"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/weights/Initializer/random_uniform/min"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: -0.06123724207282066
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/weights/Initializer/random_uniform/max"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.06123724207282066
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/weights/Initializer/random_uniform/RandomUniform"
  op: "RandomUniform"
  input: "ssd300/Conv_1/weights/Initializer/random_uniform/shape"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "seed"
    value {
      i: 0
    }
  }
  attr {
    key: "seed2"
    value {
      i: 0
    }
  }
}
node {
  name: "ssd300/Conv_1/weights/Initializer/random_uniform/sub"
  op: "Sub"
  input: "ssd300/Conv_1/weights/Initializer/random_uniform/max"
  input: "ssd300/Conv_1/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/weights/Initializer/random_uniform/mul"
  op: "Mul"
  input: "ssd300/Conv_1/weights/Initializer/random_uniform/RandomUniform"
  input: "ssd300/Conv_1/weights/Initializer/random_uniform/sub"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/weights/Initializer/random_uniform"
  op: "Add"
  input: "ssd300/Conv_1/weights/Initializer/random_uniform/mul"
  input: "ssd300/Conv_1/weights/Initializer/random_uniform/min"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/weights"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 1
        }
        dim {
          size: 320
        }
        dim {
          size: 1280
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/Conv_1/weights/Assign"
  op: "Assign"
  input: "ssd300/Conv_1/weights"
  input: "ssd300/Conv_1/weights/Initializer/random_uniform"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv_1/weights/read"
  op: "Identity"
  input: "ssd300/Conv_1/weights"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/weights"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/layer_conv2d/Conv2D"
  op: "Conv2D"
  input: "ssd300/expanded_conv_16/project/BatchNorm/batchnorm_1/Add_1"
  input: "ssd300/Conv_1/weights/read"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NCHW"
    }
  }
  attr {
    key: "dilations"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "explicit_paddings"
    value {
      list {
      }
    }
  }
  attr {
    key: "padding"
    value {
      s: "SAME"
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
  attr {
    key: "use_cudnn_on_gpu"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/beta/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1280
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/beta/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/beta/Initializer/zeros"
  op: "Fill"
  input: "ssd300/Conv_1/BatchNorm/beta/Initializer/zeros/shape_as_tensor"
  input: "ssd300/Conv_1/BatchNorm/beta/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/beta"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1280
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/beta/Assign"
  op: "Assign"
  input: "ssd300/Conv_1/BatchNorm/beta"
  input: "ssd300/Conv_1/BatchNorm/beta/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/beta"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/beta/read"
  op: "Identity"
  input: "ssd300/Conv_1/BatchNorm/beta"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/beta"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1280
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_mean/Initializer/zeros/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_mean/Initializer/zeros"
  op: "Fill"
  input: "ssd300/Conv_1/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor"
  input: "ssd300/Conv_1/BatchNorm/moving_mean/Initializer/zeros/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_mean"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1280
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_mean/Assign"
  op: "Assign"
  input: "ssd300/Conv_1/BatchNorm/moving_mean"
  input: "ssd300/Conv_1/BatchNorm/moving_mean/Initializer/zeros"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_mean"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_mean/read"
  op: "Identity"
  input: "ssd300/Conv_1/BatchNorm/moving_mean"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_mean"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
        }
        int_val: 1280
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_variance/Initializer/ones/Const"
  op: "Const"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_variance/Initializer/ones"
  op: "Fill"
  input: "ssd300/Conv_1/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor"
  input: "ssd300/Conv_1/BatchNorm/moving_variance/Initializer/ones/Const"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "index_type"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_variance"
  op: "VariableV2"
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "container"
    value {
      s: ""
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1280
        }
      }
    }
  }
  attr {
    key: "shared_name"
    value {
      s: ""
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_variance/Assign"
  op: "Assign"
  input: "ssd300/Conv_1/BatchNorm/moving_variance"
  input: "ssd300/Conv_1/BatchNorm/moving_variance/Initializer/ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_variance"
      }
    }
  }
  attr {
    key: "use_locking"
    value {
      b: true
    }
  }
  attr {
    key: "validate_shape"
    value {
      b: true
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/moving_variance/read"
  op: "Identity"
  input: "ssd300/Conv_1/BatchNorm/moving_variance"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@ssd300/Conv_1/BatchNorm/moving_variance"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/Reshape/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\000\005\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/Reshape"
  op: "Reshape"
  input: "ssd300/Conv_1/BatchNorm/beta/read"
  input: "ssd300/Conv_1/BatchNorm/Reshape/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/Reshape_1/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\000\005\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/Reshape_1"
  op: "Reshape"
  input: "ssd300/Conv_1/BatchNorm/moving_mean/read"
  input: "ssd300/Conv_1/BatchNorm/Reshape_1/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/Reshape_2/shape"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 4
          }
        }
        tensor_content: "\001\000\000\000\000\005\000\000\001\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/Reshape_2"
  op: "Reshape"
  input: "ssd300/Conv_1/BatchNorm/moving_variance/read"
  input: "ssd300/Conv_1/BatchNorm/Reshape_2/shape"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tshape"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "ssd300/Conv_1/BatchNorm/batchnorm_1/Add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
    }
  }
}