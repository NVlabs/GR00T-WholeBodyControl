import collections
from copy import copy

import numpy as np

from robocasa.models.objects.objects import MJCFObject
from robosuite.models.objects import MujocoObject
from robosuite.utils import RandomizationError
from robosuite.utils.transform_utils import (
    convert_quat,
    euler2mat,
    mat2quat,
    quat_multiply,
    rotate_2d_point,
)

from robocasa.utils.object_utils import (
    obj_in_region,
    objs_intersect,
    obj_in_region_with_keypoints,
)


class ObjectPositionSampler:
    """
    Base class of object placement sampler.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        ensure_object_boundary_in_range (bool): If True, will ensure that the object is enclosed within a given boundary
            (should be implemented by subclass)

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        reference_rot=0,
        z_offset=0.0,
        rng=None,
    ):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        # Setup attributes
        self.name = name
        if mujoco_objects is None:
            self.mujoco_objects = []
        else:
            # Shallow copy the list so we don't modify the inputted list but still keep the object references
            self.mujoco_objects = (
                [mujoco_objects]
                if isinstance(mujoco_objects, MujocoObject)
                else copy(mujoco_objects)
            )
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.ensure_valid_placement = ensure_valid_placement
        self.reference_pos = reference_pos
        self.reference_rot = reference_rot
        self.z_offset = z_offset

    def add_objects(self, mujoco_objects):
        """
        Add additional objects to this sampler. Checks to make sure there's no identical objects already stored.

        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): single model or list of MJCF object models
        """
        mujoco_objects = (
            [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects
        )
        for obj in mujoco_objects:
            assert obj not in self.mujoco_objects, "Object '{}' already in sampler!".format(
                obj.name
            )
            self.mujoco_objects.append(obj)

    def reset(self):
        """
        Resets this sampler. Removes all mujoco objects from this sampler.
        """
        self.mujoco_objects = []

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample on a surface (not necessarily table surface).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object.

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        """
        raise NotImplementedError

    @property
    def sides_combinations(self):
        return {
            "left": ["front_left", "back_left"],
            "right": ["front_right", "back_right"],
            "front": ["front_left", "front_right"],
            "back": ["back_left", "back_right"],
            "all": ["front_left", "front_right", "back_left", "back_right"],
        }

    @property
    def valid_sides(self):
        return set(
            [
                "left",
                "right",
                "front",
                "back",
                "all",
                "front_left",
                "front_right",
                "back_left",
                "back_right",
            ]
        )


class UniformRandomSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects

        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects

        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.

        num_attempts (int): Number of attempts to sample an object before giving up.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_range=(0, 0),
        y_range=(0, 0),
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        ensure_object_in_ref_region=False,
        ensure_object_out_of_ref_region=False,
        reference_pos=(0, 0, 0),
        reference_rot=0,
        z_offset=0.0,
        rng=None,
        side="all",
        num_attempts=5000,
    ):
        """Uniformly sample the position of the object.

        Args:
            ensure_object_in_ref_region (bool, optional):
                If True, we need to ensure the object be placed within the reference region. Not need to be fully in the region, but at least the center of the object.
            ensure_object_out_of_ref_region (bool, optional):
                If True, we need to ensure the object be placed outside the reference region. Need to be fully outside the region.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.num_attempts = num_attempts
        if side not in self.valid_sides:
            raise ValueError("Invalid value for side, must be one of:", self.valid_sides)

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            reference_rot=reference_rot,
            z_offset=z_offset,
            rng=rng,
        )
        self.ensure_object_in_ref_region = ensure_object_in_ref_region
        self.ensure_object_out_of_ref_region = ensure_object_out_of_ref_region

    def _sample_x(self):
        """
        Samples the x location for a given object

        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_range
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_y(self):
        """
        Samples the y location for a given object

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_range
        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object

        Returns:
            np.array: sampled object quaternion in (w,x,y,z) form

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = self.rng.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            if isinstance(self.rotation[0], collections.abc.Iterable):
                rotation = self.rng.choice(self.rotation)
            else:
                rotation = self.rotation
            rot_angle = self.rng.uniform(high=max(rotation), low=min(rotation))
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )

    def sample(self, placed_objects=None, reference=None, neg_reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            placed_objects (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str, tuple, or None, optional): Defines the reference for relative placement. It can be an object
                name found in `fixtures`, a tuple specifying an object and its site ID, or a direct (x, y, z) coordinate.
                If not provided, placement is sampled relative to this sampler's `reference_pos`.

            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if placed_objects is None else copy(placed_objects)
        spawn_ref_obj = None

        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        # Handle shelf object references
        # If the reference is provided as a tuple (object_key: str, spawn_id: int),
        # then we treat it as a reference to a specific shelf level
        elif type(reference) is tuple and len(reference) == 2:
            (reference, spawn_id) = reference
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            assert isinstance(
                ref_obj, MJCFObject
            ), "Invalid reference received. Should be of type MJCFObject"
            if spawn_id == -1:
                spawn_id, site = ref_obj.get_random_spawn(self.rng, exclude_disabled=True)
            else:
                site = ref_obj.spawns[spawn_id]
            ref_obj.set_spawn_active(spawn_id, False)

            base_offset = np.array(ref_pos)
            base_offset += ref_obj.get_spawn_bottom_offset(site)
            spawn_ref_obj = ref_obj
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert obj.name not in placed_objects, "Object '{}' has already been sampled!".format(
                obj.name
            )

            success = False

            # get reference rotation
            ref_quat = convert_quat(mat2quat(euler2mat([0, 0, self.reference_rot])), to="wxyz")

            ### get boundary points ###
            region_points = np.array(
                [
                    [self.x_range[0], self.y_range[0], 0],
                    [self.x_range[1], self.y_range[0], 0],
                    [self.x_range[0], self.y_range[1], 0],
                ]
            )
            for i in range(len(region_points)):
                region_points[i][0:2] = rotate_2d_point(
                    region_points[i][0:2], rot=self.reference_rot
                )
            region_points += base_offset

            for i in range(self.num_attempts):
                # sample object coordinates
                relative_x = self._sample_x()
                relative_y = self._sample_y()

                # apply rotation
                object_x, object_y = rotate_2d_point(
                    [relative_x, relative_y], rot=self.reference_rot
                )

                object_x = object_x + base_offset[0]
                object_y = object_y + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= obj.bottom_offset[-1]

                # random rotation
                quat = self._sample_quat()
                # multiply this quat by the object's initial rotation if it has the attribute specified
                if hasattr(obj, "init_quat"):
                    quat = quat_multiply(quat, obj.init_quat)
                quat = convert_quat(
                    quat_multiply(
                        convert_quat(ref_quat, to="xyzw"),
                        convert_quat(quat, to="xyzw"),
                    ),
                    to="wxyz",
                )

                location_valid = True

                # ensure object placed fully in region
                if self.ensure_object_boundary_in_range and not obj_in_region(
                    obj,
                    obj_pos=[object_x, object_y, object_z],
                    obj_quat=convert_quat(quat, to="xyzw"),
                    p0=region_points[0],
                    px=region_points[1],
                    py=region_points[2],
                ):
                    location_valid = False
                    continue

                # objects cannot overlap
                if self.ensure_valid_placement:
                    for (x, y, z), other_quat, other_obj in placed_objects.values():
                        if spawn_ref_obj and other_obj == spawn_ref_obj:
                            if not objs_intersect(
                                obj=obj,
                                obj_pos=[object_x, object_y, object_z],
                                obj_quat=convert_quat(quat, to="xyzw"),
                                other_obj=other_obj,
                                other_obj_pos=[x, y, z],
                                other_obj_quat=convert_quat(other_quat, to="xyzw"),
                            ):
                                location_valid = False
                                break
                        else:
                            if objs_intersect(
                                obj=obj,
                                obj_pos=[object_x, object_y, object_z],
                                obj_quat=convert_quat(quat, to="xyzw"),
                                other_obj=other_obj,
                                other_obj_pos=[x, y, z],
                                other_obj_quat=convert_quat(other_quat, to="xyzw"),
                            ):
                                location_valid = False
                                break

                # ensure at least one point of the object should be in the region
                if self.ensure_object_in_ref_region and reference is not None:
                    _ref_pos, _ref_quat, _ref_obj = placed_objects[reference]
                    if not obj_in_region_with_keypoints(
                        obj=obj,
                        obj_pos=[object_x, object_y, object_z],
                        obj_quat=convert_quat(quat, to="xyzw"),
                        region_points=_ref_obj.get_bbox_points(trans=_ref_pos, rot=_ref_quat)[:3],
                        min_num_points=3,
                    ):
                        location_valid = False
                        break

                if self.ensure_object_out_of_ref_region and neg_reference is not None:
                    for nf in neg_reference:
                        assert (
                            nf in placed_objects
                        ), "Invalid negative reference received. Current options are: {}, requested: {}".format(
                            placed_objects.keys(), nf
                        )
                        _ref_pos, _ref_quat, _ref_obj = placed_objects[nf]
                        if obj_in_region(
                            obj=obj,
                            obj_pos=[object_x, object_y, object_z],
                            obj_quat=convert_quat(quat, to="xyzw"),
                            p0=_ref_obj.get_bbox_points(trans=_ref_pos, rot=_ref_quat)[0],
                            px=_ref_obj.get_bbox_points(trans=_ref_pos, rot=_ref_quat)[1],
                            py=_ref_obj.get_bbox_points(trans=_ref_pos, rot=_ref_quat)[2],
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError(f"Cannot place object: {obj.name}")

        return placed_objects


class SequentialCompositeSampler(ObjectPositionSampler):
    """
    Samples position for each object sequentially. Allows chaining
    multiple placement initializers together - so that object locations can
    be sampled on top of other objects or relative to other object placements.

    Args:
        name (str): Name of this sampler.
    """

    def __init__(self, name, rng=None):
        # Samplers / args will be filled in later
        self.samplers = collections.OrderedDict()
        self.sample_args = collections.OrderedDict()
        self.sampler_optional = collections.OrderedDict()

        super().__init__(name=name, rng=rng)

    def append_sampler(self, sampler, sample_args=None, optional=False):
        """
        Adds a new placement initializer with corresponding @sampler and arguments

        Args:
            sampler (ObjectPositionSampler): sampler to add
            sample_args (None or dict): If specified, should be additional arguments to pass to @sampler's sample()
                call. Should map corresponding sampler's arguments to values (excluding @fixtures argument)
            optional (bool): If True, do not raise an error if the sampler fails to place all objects. This parameter
                is for handling optional clutter generation. When True, clutter objects that can't be placed due to
                space constraints are silently skipped.

        Raises:
            AssertionError: [Object name in samplers]
        """
        # Verify that all added mujoco objects haven't already been added, and add to this sampler's objects dict
        for obj in sampler.mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        self.samplers[sampler.name] = sampler
        self.sample_args[sampler.name] = sample_args
        self.sampler_optional[sampler.name] = optional

    def hide(self, mujoco_objects):
        """
        Helper method to remove an object from the workspace.

        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to hide
        """
        sampler = UniformRandomSampler(
            name="HideSampler",
            mujoco_objects=mujoco_objects,
            x_range=[-10, -20],
            y_range=[-10, -20],
            rotation=[0, 0],
            rotation_axis="z",
            z_offset=10,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
            rng=self.rng,
        )
        self.append_sampler(sampler=sampler)

    def add_objects(self, mujoco_objects):
        """
        Override super method to make sure user doesn't call this (all objects should implicitly belong to sub-samplers)
        """
        raise AttributeError("add_objects() should not be called for SequentialCompsiteSamplers!")

    def add_objects_to_sampler(self, sampler_name, mujoco_objects):
        """
        Adds specified @mujoco_objects to sub-sampler with specified @sampler_name.

        Args:
            sampler_name (str): Existing sub-sampler name
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to add
        """
        # First verify that all mujoco objects haven't already been added, and add to this sampler's objects dict
        mujoco_objects = (
            [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects
        )
        for obj in mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        # Make sure sampler_name exists
        assert (
            sampler_name in self.samplers.keys()
        ), "Invalid sub-sampler specified, valid options are: {}, " "requested: {}".format(
            self.samplers.keys(), sampler_name
        )
        # Add the mujoco objects to the requested sub-sampler
        self.samplers[sampler_name].add_objects(mujoco_objects)

    def reset(self):
        """
        Resets this sampler. In addition to base method, iterates over all sub-samplers and resets them
        """
        super().reset()
        for sampler in self.samplers.values():
            sampler.reset()

    def sample(self, placed_objects=None, reference=None, on_top=True):
        """
        Sample from each placement initializer sequentially, in the order
        that they were appended.

        Args:
            placed_objects (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. This will override each
                sampler's @reference argument if not already specified. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object. This will override each
                sampler's @on_top argument if not already specified. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
        """
        # Standardize inputs
        placed_objects = {} if placed_objects is None else copy(placed_objects)

        # Iterate through all samplers to sample
        for sampler, s_args in zip(self.samplers.values(), self.sample_args.values()):
            # Pre-process sampler args
            if s_args is None:
                s_args = {}
            for arg_name, arg in zip(("reference", "on_top"), (reference, on_top)):
                if arg_name not in s_args:
                    s_args[arg_name] = arg
            # Run sampler
            try:
                new_placements = sampler.sample(
                    placed_objects=placed_objects or s_args.pop("placed_objects", {}),
                    **s_args,
                )
            except RandomizationError:
                if self.sampler_optional[sampler.name]:
                    continue
                else:
                    raise
            # Update placements
            placed_objects.update(new_placements)

        # only return placements for newly placed objects
        sampled_obj_names = [
            obj.name for sampler in self.samplers.values() for obj in sampler.mujoco_objects
        ]
        return {k: v for (k, v) in placed_objects.items() if k in sampled_obj_names}


class MultiRegionSampler(ObjectPositionSampler):
    def __init__(
        self,
        name,
        regions,
        side="all",
        mujoco_objects=None,
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        ensure_object_in_ref_region=False,
        rng=None,
        z_offset=0.0,
    ):
        if len(regions) != 4:
            raise ValueError("Exactly four sites (one for each quadrant) must be provided.")
        if side not in self.valid_sides:
            raise ValueError("Invalid value for side, must be one of:", self.valid_sides)

        # initialize sides and regions
        if side in self.sides_combinations:
            self.sides = self.sides_combinations[side]
        else:
            self.sides = [side]
        self.regions = regions

        # create a list of uniform samplers (one for each site)
        self.samplers = list()
        for s in self.sides:
            site = self.regions[s]
            sampler = UniformRandomSampler(
                name=name,
                mujoco_objects=mujoco_objects,
                reference_pos=site["pos"],
                x_range=site["x_range"],
                y_range=site["y_range"],
                rotation=rotation,
                rotation_axis=rotation_axis,
                ensure_object_boundary_in_range=ensure_object_boundary_in_range,
                ensure_valid_placement=ensure_valid_placement,
                ensure_object_in_ref_region=ensure_object_in_ref_region,
                z_offset=z_offset,
                rng=rng,
            )
            self.samplers.append(sampler)

    def sample(self, fixtures=None, reference=None, on_top=True):
        # randomly picks a sampler and calls its sample function
        sampler = self.rng.choice(self.samplers)
        return sampler.sample(fixtures=fixtures, reference=reference, on_top=on_top)
