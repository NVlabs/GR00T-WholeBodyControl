GR00T-WholeBodyControl Documentation
====================================

.. image:: https://img.shields.io/badge/License-Apache%202.0%20%7C%20NVIDIA%20Open%20Model-blue.svg
   :target: resources/license.html
   :alt: License

.. image:: https://img.shields.io/badge/IsaacLab-2.3.0-blue.svg
   :target: https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.0
   :alt: IsaacLab

Welcome to the official documentation for **GR00T Whole-Body Control (WBC)**! This is a unified platform for developing and deploying advanced humanoid controllers.


What is GR00T-WholeBodyControl?
--------------------------------

This codebase serves as the foundation for:

- **Decoupled WBC** models used in NVIDIA Isaac-Gr00t, Gr00t N1.5 and N1.6 (see :doc:`detailed reference <references/decoupled_wbc>`)
- **GEAR-SONIC Series**: State-of-the-art controllers from the GEAR team

GEAR-SONIC
----------
.. image:: _static/sonic-preview-gif-480P.gif
   :width: 100%
   :align: center


.. raw:: html

   <p style="margin-top: 0; margin-bottom: 1em;">
     <a href="https://nvlabs.github.io/GEAR-SONIC/"><img src="https://img.shields.io/badge/🌐_Website-GEAR--SONIC-76B900" alt="Website"></a>
     <a href="https://arxiv.org/abs/2511.07820"><img src="https://img.shields.io/badge/📄_arXiv-2511.07820-b31b1b" alt="Paper"></a>
     <a href="https://github.com/NVlabs/GR00T-WholeBodyControl"><img src="https://img.shields.io/badge/💻_GitHub-Repository-181717" alt="GitHub"></a>
   </p>

**SONIC** is a humanoid behavior foundation model that gives robots a core set of motor skills learned from large-scale human motion data. Rather than building separate controllers for every motion, SONIC uses motion tracking as a scalable training task so a single unified policy can produce natural, whole-body movement and support a wide range of behaviors.

🎯 Key Features:

- 🚶 Natural whole-body locomotion (walking, crawling, dynamic movements)
- 🎮 Real-time VR teleoperation support
- 🤖 Foundation for higher-level planning and interaction
- 📦 Ready-to-deploy C++ inference stack

Quick Start: Sim2Sim
--------------------

Quickly test the SONIC deployment stack in MuJoCo before deploying on real hardware.

.. raw:: html

   <video width="100%" autoplay loop muted playsinline style="border-radius: 8px; margin: 0 0 1.5em 0;">
     <source src="_static/sim2sim.mp4" type="video/mp4">
   </video>

.. tip::

   **Get running in minutes!** Follow the :doc:`Installation <getting_started/installation_deploy>` and :doc:`Quickstart <getting_started/quickstart>` guides to see this in action on your machine.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation_deploy
   getting_started/download_models
   getting_started/quickstart
   getting_started/vr_teleop_setup
..    getting_started/installation_training

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/keyboard
   tutorials/gamepad
   tutorials/zmq
   tutorials/manager
   tutorials/vr_wholebody_teleop

.. toctree::
   :maxdepth: 2
   :caption: Best Practices

   user_guide/teleoperation

..    user_guide/configuration
..    user_guide/training
..    user_guide/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: API Reference

..    api/index
..    api/teleop

.. toctree::
   :maxdepth: 2
   :caption: Reference Documentation

   references/index
   references/deployment_code
   references/observation_config
   references/motion_reference
   references/planner_onnx
   references/jetpack6
   references/decoupled_wbc
   

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   resources/citations
   resources/license
   resources/support
..    resources/contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
