from __future__ import annotations

from .src.utils import export_tree as _export_tree
from ._pywrap import BaseTree, Classifier, Regressor
import json
from typing import Optional, Dict, Any, List, Union
from io import BytesIO
import os
import numpy as np


def load_tree(tree_data: Union[str, Dict]) -> Union[Classifier, Regressor]:
    """
    Load a decision tree model from a JSON file or dictionary representation.

    This function reconstructs a trained decision tree from its serialized form,
    either from a JSON file on disk or a dictionary containing the tree structure
    and parameters.

    Parameters
    ----------
    tree_data : Union[str, Dict]
        Either:
        - A string containing the file path to a JSON file containing the tree data
        - A dictionary containing the serialized tree structure and parameters

    Returns
    -------
    Union[Classifier, Regressor]
        A reconstructed decision tree object. The specific type (Classifier or
        Regressor) is determined by the 'task' parameter in the tree data.
    """
    # Handle input types
    if isinstance(tree_data, str):
        if not os.path.exists(tree_data):
            raise FileNotFoundError(f"The file {tree_data} does not exist")

        with open(tree_data, "r") as f:
            tree = json.load(f)
    elif isinstance(tree_data, dict):
        tree = tree_data
    else:
        raise ValueError("Input must be a JSON string, file path, or dictionary")

    # Validate tree structure
    if (
        not isinstance(tree, dict)
        or "params" not in tree
        or "task" not in tree["params"]
    ):
        raise ValueError("Invalid tree data structure")

    # Create appropriate object based on task
    if not tree["params"]["task"]:
        obj = Classifier.__new__(Classifier)
    else:
        obj = Regressor.__new__(Regressor)

    tree["_fit"] = True

    obj.__setstate__(tree)
    
    return obj


def export_tree(
    tree: Union[Classifier, Regressor], out_file: str = None
) -> Union[None, dict]:
    """
    Serialize a decision tree model to a dictionary or JSON file.

    This function converts a trained decision tree into a portable format that can
    be saved to disk or transmitted. The serialized format preserves all necessary
    information to reconstruct the tree using load_tree().

    Parameters
    ----------
    tree : Union[Classifier, Regressor]
        The trained decision tree model to export. Must be an instance of either
        Classifier or Regressor and must have been fitted.

    out_file : str, optional
        If provided, the path where the serialized tree should be saved as a JSON
        file. If None, the function returns the dictionary representation instead
        of saving to disk.

    Returns
    -------
    Union[None, dict]
        If out_file is None:
            Returns a dictionary containing the serialized tree structure and parameters
        If out_file is provided:
            Returns None after saving the tree to the specified JSON file
    """
    if not isinstance(tree, BaseTree):
        raise ValueError("`tree` must be an instance of `BaseTree`.")

    if not tree._fit:
        raise ValueError(
            "The tree has not been fitted yet. Please call the 'fit' method to train the tree before using this function."
        )

    tree_dict = _export_tree(tree)  # Assuming this function is implemented elsewhere.

    if out_file is not None:
        if isinstance(out_file, str):
            with open(out_file, "w") as f:
                json.dump(tree_dict, f, indent=4)
        else:
            raise ValueError("`out_file` must be a string if provided.")

    else:
        return tree_dict


def visualize_tree(
    tree: Union[Classifier, Regressor],
    feature_names: Optional[List[str]] = None,
    max_cat: Optional[int] = None,
    max_oblique: Optional[int] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    figsize: tuple = (20, 10),
    show_gini: bool = True,
    show_n_samples: bool = True,
    show_node_value: bool = True,
) -> None:
    """
    Generate a visual representation of a decision tree model.

    Creates a graphical visualization of the tree structure showing decision nodes,
    leaf nodes, and split criteria. The visualization can be customized to show
    various node statistics and can be displayed or saved to a file.

    Parameters
    ----------
    tree : Union[Classifier, Regressor]
        The trained decision tree model to visualize. Must be fitted before
        visualization.

    feature_names : List[str], optional
        Human-readable names for the features used in the tree. If provided,
        these names will be used in split conditions instead of generic feature
        indices (e.g., "age <= 30" instead of "f0 <= 30").

    max_cat : int, optional
        For categorical splits, limits the number of categories shown in the
        visualization. If there are more categories than this limit, they will
        be truncated with an ellipsis. Useful for splits with many categories.

    max_oblique : int, optional
        For oblique splits (those involving multiple features), limits the number
        of features shown in the split condition. Helps manage complex oblique
        splits in the visualization.

    save_path : str, optional
        If provided, saves the visualization to this file path. The file format
        is determined by the file extension (e.g., '.png', '.pdf').

    dpi : int, default=600
        The resolution (dots per inch) of the saved image. Only relevant if
        save_path is provided.

    figsize : tuple, default=(20, 10)
        The width and height of the figure in inches.

    show_gini : bool, default=True
        Whether to display Gini impurity values in the nodes.

    show_n_samples : bool, default=True
        Whether to display the number of samples that reach each node.

    show_node_value : bool, default=True
        Whether to display the predicted value/class distributions in each node.

    Returns
    -------
    None
        The function displays the visualization and optionally saves it to disk.
    """
    _check_visualize_tree_inputs(
        tree, feature_names, max_cat, max_oblique, save_path, dpi, figsize
    )

    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            "graphviz is not installed. Please install it to use this function."
        )

    try:
        from matplotlib.pyplot import figure, imshow, imread, axis, savefig, show
    except:
        raise ImportError(
            "matplotlib is not installed. Please install it to use this function."
        )

    tree_dict = _export_tree(tree)  # Assuming this function is implemented elsewhere.

    node, params = tree_dict["tree"], tree_dict["params"]

    def _visualize_recursive(node, graph=None, parent=None, edge_label=""):
        if graph is None:
            graph = Digraph(format="png")
            graph.graph_attr.update(
                {
                    "rankdir": "TB",
                    "ranksep": "0.3",
                    "nodesep": "0.2",
                    "splines": "polyline",
                    "ordering": "out",
                }
            )
            graph.attr(
                "node",
                shape="box",
                style="filled",
                color="lightgrey",
                fontname="Helvetica",
                margin="0.2",
            )

        node_id = str(id(node))
        label_parts = []

        is_leaf = "left" not in node and "right" not in node

        if is_leaf:
            # For leaf nodes
            label_parts.append("leaf")
            label_parts.append(
                _format_value_str(node, params, feature_names, max_oblique)
            )

            # Add impurity for leaf nodes if requested and available
            if show_gini and "impurity" in node:
                label_parts.append(f"impurity: {_format_float(node['impurity'])}")

            # Add n_samples for leaf nodes if requested and available
            if show_n_samples and "n_samples" in node:
                label_parts.append(f"n_samples: {node['n_samples']}")

            graph.node(
                node_id,
                label="\n".join(label_parts),
                shape="box",
                style="filled",
                color="lightblue",
                fontname="Helvetica",
            )
        else:
            # For internal nodes
            # First add the split information
            if node.get("is_oblique", False):
                split_info = _create_oblique_expression(
                    node["features"],
                    node["weights"],
                    node["threshold"],
                    feature_names,
                    max_oblique,
                )
            elif "category_left" in node:
                categories = node["category_left"]
                cat_str = _format_categories(categories, max_cat)
                feature_label = (
                    feature_names[node["feature_idx"]]
                    if feature_names and node["feature_idx"] < len(feature_names)
                    else f"f{node['feature_idx']}"
                )
                split_info = f"{feature_label} in {cat_str}"
            else:
                threshold = (
                    _format_float(node["threshold"])
                    if isinstance(node["threshold"], float)
                    else node["threshold"]
                )
                feature_label = (
                    feature_names[node["feature_idx"]]
                    if feature_names and node["feature_idx"] < len(feature_names)
                    else f"f{node['feature_idx']}"
                )
                split_info = f"{feature_label} ≤ {threshold}"

            label_parts.append(split_info)

            if show_node_value:
                label_parts.append(
                    _format_value_str(node, params, feature_names, max_oblique)
                )

            # Add Gini impurity if requested
            if show_gini and "impurity" in node:
                label_parts.append(f"impurity: {_format_float(node['impurity'])}")

            # Add n_samples if requested
            if show_n_samples and "n_samples" in node:
                label_parts.append(f"n_samples: {node['n_samples']}")

            graph.node(
                node_id,
                label="\n".join(label_parts),
                shape="box",
                style="filled",
                color="lightgrey",
                fontname="Helvetica",
            )

        if parent is not None:
            graph.edge(
                parent,
                node_id,
                label=edge_label,
                fontname="Helvetica",
                penwidth="1.0",
                minlen="1",
            )

        if "left" in node:
            _visualize_recursive(node["left"], graph, node_id, "Left")
        if "right" in node:
            _visualize_recursive(node["right"], graph, node_id, "Right")

        return graph

    graph = _visualize_recursive(node)
    png_data = graph.pipe(format="png")

    figure(figsize=figsize)
    imshow(imread(BytesIO(png_data)))
    axis("off")

    if save_path:
        savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    show()
    
def export_tree_to_onnx(tree: Union[Classifier, Regressor]) -> None:
    """
    Convert an oblique decision tree (Classifier or Regressor) into an ONNX model.

    .. important::
       - This implementation currently does **not** support batch processing.
         Only a single row (1D NumPy array) and np.float64 dtype can be passed as input.
       - The input variable name must be **"X"** and its shape should be (n_features,).
       - In binary classification, the output is a single-dimensional value representing 
         the probability of belonging to the positive class.

    Parameters
    ----------
    tree : Union[Classifier, Regressor]
        The oblique decision tree (classifier or regressor) to be converted to ONNX.

    Returns
    -------
    onnx.ModelProto
        The constructed ONNX model.

    Examples
    --------
    >>> # Suppose we have a 2D NumPy array X of shape (num_samples, num_features).
    >>> # We only take a single row for prediction:
    >>> X_sample = X[0, :]
    >>> 
    >>> # Create an inference session using onnxruntime:
    >>> import onnxruntime
    >>> session = onnxruntime.InferenceSession("tree.onnx")
    >>> 
    >>> # Retrieve the output name of the model
    >>> out_name = session.get_outputs()[0].name
    >>> 
    >>> # Perform inference on the sample
    >>> y_pred = session.run([out_name], {"X": X_sample})[0]
    >>> print(y_pred)
    """
    try:
        from onnx import helper, TensorProto
    except ImportError as e:
        raise ImportError(
            "Failed to import onnx dependencies. Please make sure the 'onnx' "
            "package is installed."
        ) from e
    
    tree_dict = export_tree(tree)

    # Closure for unique name generation
    name_counter = [0]

    def _unique_name(prefix="Node"):
        name_counter[0] += 1
        return f"{prefix}_{name_counter[0]}"

    def _make_constant_int_node(name, value, shape=None):
        """
        Creates an ONNX Constant node containing int64 data.
        Useful for indices in Gather or other integer-only parameters.
        """
        if shape is None:
            shape = [len(value)] if isinstance(value, list) else []
        arr = (
            np.array(value, dtype=np.int64)
            if isinstance(value, list)
            else (
                np.array([value], dtype=np.int64)
                if shape == []
                else np.array(value, dtype=np.int64)
            )
        )

        const_tensor = helper.make_tensor(
            name=_unique_name("const_data_int"),
            data_type=TensorProto.INT64,
            dims=arr.shape,
            vals=arr.flatten().tolist(),
        )

        node = helper.make_node(
            "Constant", inputs=[], outputs=[name], value=const_tensor
        )
        return node

    def _make_constant_float_node(name, value, shape=None):
        """
        Creates an ONNX Constant node containing float64 data.
        Useful for thresholds, weights, etc.
        """
        if shape is None:
            shape = [len(value)] if isinstance(value, list) else []
        arr = (
            np.array(value, dtype=np.float64)
            if isinstance(value, list)
            else np.array([value], dtype=np.float64)
        )

        if shape and arr.shape != tuple(shape):
            arr = arr.reshape(shape)

        const_tensor = helper.make_tensor(
            name=_unique_name("const_data_float"),
            data_type=TensorProto.DOUBLE,
            dims=arr.shape,
            vals=arr.flatten().tolist(),
        )
        node = helper.make_node(
            "Constant", inputs=[], outputs=[name], value=const_tensor
        )
        return node

    def _emit_linear_leaf(node_dict, nodes_list, target_name):
        """Emit ONNX ops to evaluate a linear / logistic / softmax leaf."""
        leaf_coef = node_dict["leaf_coef"]
        leaf_intercept = node_dict["leaf_intercept"]
        n_models = int(node_dict.get("leaf_n_models", 1))
        n_coef = int(node_dict.get("leaf_n_coef", 0))
        numeric_features_local = params.get("numeric_features") or []
        task_local = params["task"]
        n_classes_local = params["n_classes"]

        # Build per-model logit (each is a [1] tensor in ONNX).
        logit_outputs = []
        for k in range(n_models):
            partials = []
            for j in range(n_coef):
                feat_idx = (
                    numeric_features_local[j]
                    if j < len(numeric_features_local)
                    else j
                )
                idx_node = _make_constant_int_node(
                    _unique_name("nf_idx"), [feat_idx], [1]
                )
                nodes_list.append(idx_node)

                gout = _unique_name("nf_gather")
                nodes_list.append(
                    helper.make_node(
                        "Gather",
                        inputs=["X", idx_node.output[0]],
                        outputs=[gout],
                        axis=0,
                    )
                )

                wnode = _make_constant_float_node(
                    _unique_name("leaf_w"), leaf_coef[k * n_coef + j], []
                )
                nodes_list.append(wnode)

                mout = _unique_name("leaf_mul")
                nodes_list.append(
                    helper.make_node(
                        "Mul", inputs=[gout, wnode.output[0]], outputs=[mout]
                    )
                )
                partials.append(mout)

            inter_node = _make_constant_float_node(
                _unique_name("leaf_b"), leaf_intercept[k], []
            )
            nodes_list.append(inter_node)

            if not partials:
                logit_k = inter_node.output[0]
            else:
                tmp = partials[0]
                for p in partials[1:]:
                    aout = _unique_name("leaf_add")
                    nodes_list.append(
                        helper.make_node("Add", inputs=[tmp, p], outputs=[aout])
                    )
                    tmp = aout
                logit_out = _unique_name("leaf_logit")
                nodes_list.append(
                    helper.make_node(
                        "Add",
                        inputs=[tmp, inter_node.output[0]],
                        outputs=[logit_out],
                    )
                )
                logit_k = logit_out
            logit_outputs.append(logit_k)

        if n_models == 1:
            if task_local:
                # regression: squeeze [1] -> scalar
                nodes_list.append(
                    helper.make_node(
                        "Squeeze", inputs=[logit_outputs[0]], outputs=[target_name]
                    )
                )
            elif n_classes_local == 2:
                # binary classification: sigmoid then squeeze
                sig_out = _unique_name("leaf_sig")
                nodes_list.append(
                    helper.make_node(
                        "Sigmoid", inputs=[logit_outputs[0]], outputs=[sig_out]
                    )
                )
                nodes_list.append(
                    helper.make_node(
                        "Squeeze", inputs=[sig_out], outputs=[target_name]
                    )
                )
            else:
                nodes_list.append(
                    helper.make_node(
                        "Identity",
                        inputs=[logit_outputs[0]],
                        outputs=[target_name],
                    )
                )
        else:
            # Multiclass: concat K logits (each [1]) -> [K], then softmax along axis=0.
            cat_out = _unique_name("leaf_logits")
            nodes_list.append(
                helper.make_node(
                    "Concat", inputs=logit_outputs, outputs=[cat_out], axis=0
                )
            )
            nodes_list.append(
                helper.make_node(
                    "Softmax", inputs=[cat_out], outputs=[target_name], axis=0
                )
            )

    def _build_subgraph_for_node(node_dict, n_classes):
        """
        Recursively builds a subgraph (for 'If' branches) from the given node definition.
        The subgraph uses 'X' as an outer-scope input (not declared in inputs[]).
        """
        nodes = []
        graph_name = _unique_name("SubGraph")

        # Subgraph output
        out_name = _unique_name("sub_out")
        out_info = helper.make_tensor_value_info(out_name, TensorProto.DOUBLE, None)

        # Reference to 'X' from the outer scope
        x_info = helper.make_tensor_value_info("X", TensorProto.DOUBLE, [None])

        # If this is a leaf node
        if node_dict["is_leaf"]:
            has_linear = (
                node_dict.get("leaf_n_models", 0) > 0
                and "leaf_coef" in node_dict
                and "leaf_intercept" in node_dict
            )
            if has_linear:
                _emit_linear_leaf(node_dict, nodes, out_name)
            elif "values" in node_dict and isinstance(node_dict["values"], list):
                # Multi-class leaf
                val_array = node_dict["values"]
                shape = [len(val_array)]
                cnode = _make_constant_float_node(out_name, val_array, shape)
                nodes.append(cnode)
            else:
                # Single-value leaf (binary or regression)
                val = node_dict["value"]
                cnode = _make_constant_float_node(out_name, val, [])
                nodes.append(cnode)

            subgraph = helper.make_graph(
                nodes=nodes,
                name=graph_name,
                inputs=[],
                outputs=[out_info],
                value_info=[x_info],
            )
            return subgraph, out_name

        # Otherwise, this node is a split
        cond_name = _unique_name("cond_bool")
        is_oblique = node_dict.get("is_oblique", False)
        cat_list = node_dict.get("category_left", [])
        n_category = len(cat_list)

        # Oblique split
        if is_oblique:
            w_list = node_dict["weights"]
            f_list = node_dict["features"]
            thr_val = node_dict["threshold"]

            partials = []
            for w, f_idx in zip(w_list, f_list):
                gather_idx = _make_constant_int_node(
                    _unique_name("gather_idx"), [f_idx], [1]
                )
                nodes.append(gather_idx)

                gather_out = _unique_name("gather_out")
                gnode = helper.make_node(
                    "Gather",
                    inputs=["X", gather_idx.output[0]],
                    outputs=[gather_out],
                    axis=0,
                )
                nodes.append(gnode)

                w_node = _make_constant_float_node(_unique_name("weight"), w, [])
                nodes.append(w_node)

                mul_out = _unique_name("mul_out")
                mul_node = helper.make_node(
                    "Mul", inputs=[gather_out, w_node.output[0]], outputs=[mul_out]
                )
                nodes.append(mul_node)
                partials.append(mul_out)

            # Summation of partial products
            if len(partials) == 1:
                final_dot = partials[0]
            else:
                tmp = partials[0]
                for p in partials[1:]:
                    add_out = _unique_name("add_out")
                    add_node = helper.make_node(
                        "Add", inputs=[tmp, p], outputs=[add_out]
                    )
                    nodes.append(add_node)
                    tmp = add_out
                final_dot = tmp

            thr_node = _make_constant_float_node(_unique_name("thr"), thr_val, [])
            nodes.append(thr_node)

            less_node = helper.make_node(
                "Less", inputs=[final_dot, thr_node.output[0]], outputs=[cond_name]
            )
            nodes.append(less_node)

        # Categorical split
        elif n_category > 0:
            f_idx = node_dict["feature_idx"]
            fnode = _make_constant_int_node(_unique_name("catf_idx"), [f_idx], [1])
            nodes.append(fnode)

            gout = _unique_name("cat_gather_out")
            gnode = helper.make_node(
                "Gather", inputs=["X", fnode.output[0]], outputs=[gout], axis=0
            )
            nodes.append(gnode)

            eq_outputs = []
            for c_val in cat_list:
                cat_node = _make_constant_float_node(_unique_name("cat_val"), c_val, [])
                nodes.append(cat_node)

                eq_out = _unique_name("eq_out")
                eq_node = helper.make_node(
                    "Equal", inputs=[gout, cat_node.output[0]], outputs=[eq_out]
                )
                nodes.append(eq_node)
                eq_outputs.append(eq_out)

            if len(eq_outputs) == 1:
                final_eq = eq_outputs[0]
            else:
                tmp = eq_outputs[0]
                for eqo in eq_outputs[1:]:
                    or_out = _unique_name("or_out")
                    or_node = helper.make_node(
                        "Or", inputs=[tmp, eqo], outputs=[or_out]
                    )
                    nodes.append(or_node)
                    tmp = or_out
                final_eq = tmp

            id_node = helper.make_node(
                "Identity", inputs=[final_eq], outputs=[cond_name]
            )
            nodes.append(id_node)

        # Axis-aligned numeric split
        else:
            f_idx = node_dict["feature_idx"]
            thr_val = node_dict["threshold"]

            fnode = _make_constant_int_node(_unique_name("f_idx"), [f_idx], [1])
            nodes.append(fnode)

            gout = _unique_name("gather_out")
            gnode = helper.make_node(
                "Gather", inputs=["X", fnode.output[0]], outputs=[gout], axis=0
            )
            nodes.append(gnode)

            thr_node = _make_constant_float_node(_unique_name("thr_val"), thr_val, [])
            nodes.append(thr_node)

            less_node = helper.make_node(
                "Less", inputs=[gout, thr_node.output[0]], outputs=[cond_name]
            )
            nodes.append(less_node)

        # Recursively build subgraphs for left and right
        left_sub, left_out = _build_subgraph_for_node(node_dict["left"], n_classes)
        right_sub, right_out = _build_subgraph_for_node(node_dict["right"], n_classes)

        if_out = _unique_name("if_out")
        if_info = helper.make_tensor_value_info(if_out, TensorProto.DOUBLE, None)

        if_node = helper.make_node(
            "If",
            inputs=[cond_name],
            outputs=[if_out],
            name=_unique_name("IfNode"),
            then_branch=left_sub,
            else_branch=right_sub,
        )
        nodes.append(if_node)

        subgraph = helper.make_graph(
            nodes=nodes,
            name=graph_name,
            inputs=[],
            outputs=[if_info],
            value_info=[x_info],
        )
        return subgraph, if_out

    # Retrieve tree parameters
    params = tree_dict["params"]
    n_classes = params.get("n_classes", 2)
    n_features = params.get("n_features", 4)

    # Build the root subgraph from the tree
    root_subgraph, root_out_name = _build_subgraph_for_node(
        tree_dict["tree"], n_classes
    )

    # Main graph I/O
    main_input = helper.make_tensor_value_info("X", TensorProto.DOUBLE, [n_features])
    main_output = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, None)

    # Extract nodes and value_info from the root subgraph
    nodes = list(root_subgraph.node)
    val_info = list(root_subgraph.value_info)
    if_out_name = root_subgraph.output[0].name

    # Add a final Identity node to map subgraph output to "Y"
    final_out_node_name = _unique_name("final_y")
    identity_node = helper.make_node(
        "Identity", inputs=[if_out_name], outputs=[final_out_node_name]
    )
    nodes.append(identity_node)
    main_output.name = final_out_node_name

    # Construct the main graph
    main_graph = helper.make_graph(
        nodes=nodes,
        name="MainGraph",
        inputs=[main_input],
        outputs=[main_output],
        value_info=val_info,
    )

    # Fix output shape to [1] or [n_classes]
    if n_classes > 2:
        dim = main_graph.output[0].type.tensor_type.shape.dim.add()
        dim.dim_value = n_classes
    else:
        dim = main_graph.output[0].type.tensor_type.shape.dim.add()
        dim.dim_value = 1

    # Fix input shape to [n_features]
    main_graph.input[0].type.tensor_type.shape.dim[0].dim_value = n_features

    onnx_model = helper.make_model(
        main_graph,
        producer_name="custom_oblique_categorical_tree",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx_model.ir_version = 7

    return onnx_model


def _format_float(value: float) -> str:
    """Format float value with 3 decimal places, return '0' for 0.0"""
    if value == 0.0:
        return "0"
    return f"{value:.2f}"


def _format_value_str(
    node: Dict[str, Any],
    params: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    max_terms: Optional[int] = None,
) -> str:
    """
    Format value string based on task type (regression vs classification) and number of classes.

    For linear leaves (``linear_leaf=True``) on a leaf node, the formula
    ``intercept + sum(coef * feat)`` is rendered (with ``sigmoid`` for binary
    and ``softmax`` for multiclass). Falls back to the constant leaf format
    otherwise.

    Parameters:
    -----------
    node : Dict[str, Any]
        The tree node dictionary containing values or value
    params : Dict[str, Any]
        Tree parameters containing task and n_classes information
    feature_names : Optional[List[str]]
        Optional human-readable feature names used inside the linear-leaf formula.
    max_terms : Optional[int]
        Optional limit on the number of feature terms to render per linear leaf.
    """
    is_leaf = "left" not in node and "right" not in node
    if (
        is_leaf
        and node.get("leaf_n_models", 0) > 0
        and "leaf_coef" in node
        and "leaf_intercept" in node
    ):
        return _format_linear_leaf_str(node, params, feature_names, max_terms)

    # Check if it's a regression task
    if params["task"]:
        value = (
            _format_float(node["value"])
            if isinstance(node["value"], float)
            else node["value"]
        )
        return f"Value: {value}"

    # Classification task
    else:
        if params["n_classes"] == 2:  # Binary classification
            # For binary case, node["values"] contains probability for positive class
            p = node["value"]
            return f"values: [{_format_float(1-p)}, {_format_float(p)}]"
        else:  # Multiclass (3 or more classes)
            values_str = ", ".join(
                _format_float(v) if isinstance(v, float) else str(v)
                for v in node.get("values", [0.0] * params["n_classes"])
            )
            return f"values: [{values_str}]"


def _format_linear_leaf_str(
    node: Dict[str, Any],
    params: Dict[str, Any],
    feature_names: Optional[List[str]],
    max_terms: Optional[int],
) -> str:
    """Render a linear-leaf formula for visualization."""
    coef = node["leaf_coef"]
    intercept = node["leaf_intercept"]
    n_models = int(node.get("leaf_n_models", 1))
    n_coef = int(node.get("leaf_n_coef", 0))
    numeric_features = params.get("numeric_features") or list(range(n_coef))
    task = params["task"]
    n_classes = params["n_classes"]

    def feat_label(j: int) -> str:
        idx = numeric_features[j] if j < len(numeric_features) else j
        if feature_names and idx < len(feature_names):
            return feature_names[idx]
        return f"f{idx}"

    def fmt_linear(model_idx: int) -> str:
        b = intercept[model_idx]
        coefs = [coef[model_idx * n_coef + j] for j in range(n_coef)]

        order = sorted(range(n_coef), key=lambda j: abs(coefs[j]), reverse=True)
        truncated = False
        if max_terms is not None and len(order) > max_terms:
            order = order[:max_terms]
            truncated = True

        pieces = [_format_float(b)]
        for j in order:
            w = coefs[j]
            if w == 0.0:
                continue
            sign = "-" if w < 0 else "+"
            pieces.append(f"{sign} {_format_float(abs(w))}·{feat_label(j)}")
        if truncated:
            pieces.append("+ ...")
        return " ".join(pieces)

    if n_models == 1:
        formula = fmt_linear(0)
        if task:  # regression
            return f"y = {formula}"
        if n_classes == 2:  # binary
            return f"P(y=1) = σ({formula})"
        return f"linear: {formula}"  # shouldn't normally reach (single-model multiclass)

    # multiclass softmax
    lines = [f"linear leaf · softmax over {n_models} classes:"]
    for k in range(n_models):
        lines.append(f"  z[{k}] = {fmt_linear(k)}")
    return "\n".join(lines)


def _create_oblique_expression(
    features: list,
    weights: list,
    threshold: float,
    feature_names: Optional[List[str]],
    max_oblique: Optional[int] = None,
) -> str:
    """Create mathematical expression for oblique split with line breaks after 5 terms"""
    terms = []

    # Sort features and weights by absolute weight value
    feature_weight_pairs = sorted(
        zip(features, weights), key=lambda x: abs(x[1]), reverse=True
    )

    # Apply max_oblique limit if specified
    if max_oblique is not None:
        feature_weight_pairs = feature_weight_pairs[:max_oblique]
        if len(features) > max_oblique:
            feature_weight_pairs.append(("...", 0))

    # Create terms with proper formatting
    lines = []
    current_line = []

    for i, (f, w) in enumerate(feature_weight_pairs):
        if f == "...":
            current_line.append("...")
            continue

        feature_label = (
            feature_names[f] if feature_names and f < len(feature_names) else f"f{f}"
        )

        if w == 1.0:
            term = feature_label  # Removed parentheses for coefficient 1
        elif w == -1.0:
            term = f"–{feature_label}"  # Removed parentheses for coefficient -1
        else:
            formatted_weight = _format_float(abs(w))
            term = f"{'– ' if w < 0 else ''}({formatted_weight} * {feature_label})"

        if i > 0:
            term = f"+ {term}" if w > 0 else f" {term}"

        current_line.append(term)

        # Start new line after every 5 terms
        if len(current_line) == 5 and i < len(feature_weight_pairs) - 1:
            lines.append(" ".join(current_line) + " +")
            current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    formatted_threshold = _format_float(threshold)
    expression = "\n".join(lines)
    return f"{expression} ≤ {formatted_threshold}"


def _format_categories(categories: list, max_cat: Optional[int] = None) -> str:
    """Format category list with line breaks after every 5 items"""
    if max_cat is not None and len(categories) > max_cat:
        shown_cats = categories[:max_cat]
        return f"[{', '.join(map(str, shown_cats))}, ...]"

    formatted_cats = []
    current_line = []

    for i, cat in enumerate(categories):
        current_line.append(str(cat))

        # Add line break after every 5 items or at the end
        if len(current_line) == 9 and i < len(categories) - 1:
            formatted_cats.append(", ".join(current_line) + ",")
            current_line = []

    if current_line:
        formatted_cats.append(", ".join(current_line))

    if len(formatted_cats) > 1:
        return "[" + "\n".join(formatted_cats) + "]"
    return f"[{formatted_cats[0]}]"


def _check_visualize_tree_inputs(
    tree: BaseTree,
    feature_names: Optional[List[str]] = None,
    max_cat: Optional[int] = None,
    max_oblique: Optional[int] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    figsize: tuple = (20, 10),
) -> None:
    """
    Validate the inputs for the visualize_tree function.

    Parameters:
    -----------
    tree : object
        The tree object to be visualized, must have a certain expected structure.
    feature_names : Optional[List[str]]
        If provided, must be a list of strings matching the number of features in the tree.
    max_cat : Optional[int]
        If provided, must be a positive integer.
    max_oblique : Optional[int]
        If provided, must be a positive integer.
    save_path : Optional[str]
        If provided, must be a valid file path ending in a supported image format (e.g., '.png').
    dpi : int
        Must be a positive integer.
    figsize : tuple
        Must be a tuple of two positive numbers.
    """
    if not isinstance(tree, BaseTree):
        raise ValueError("`tree` must be an instance of `BaseTree`.")

    if not tree._fit:
        raise ValueError(
            "The tree has not been fitted yet. Please call the 'fit' method to train the tree before using this function."
        )

    if feature_names is not None:
        if not isinstance(feature_names, list) or not all(
            isinstance(f, str) for f in feature_names
        ):
            raise ValueError("feature_names must be a list of strings.")
        if len(feature_names) != tree.n_features:
            raise ValueError(
                f"feature_names must match the number of features in the tree ({tree.n_features})."
            )

    if max_cat is not None and (not isinstance(max_cat, int) or max_cat <= 0):
        raise ValueError("max_cat must be a positive integer.")

    if max_oblique is not None and (
        not isinstance(max_oblique, int) or max_oblique <= 0
    ):
        raise ValueError("max_oblique must be a positive integer.")

    if save_path is not None and not isinstance(save_path, str):
        raise ValueError("save_path must be a string representing a valid file path.")

    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    if (
        not isinstance(figsize, tuple)
        or len(figsize) != 2
        or not all(isinstance(dim, (int, float)) and dim > 0 for dim in figsize)
    ):
        raise ValueError("figsize must be a tuple of two positive numbers.")
