.. meta::
  :description: Composable Kernel documentation and API reference library
  :keywords: composable kernel, CK, ROCm, API, documentation

.. _contributing-to:

********************************************************************
Contributing to Composable Kernel
********************************************************************

Please read the `Composable Kernel documentation <https://rocm.docs.amd.com/projects/composable_kernel/en/latest/>`_ before contributing to the Composable Kernel project. The Composable Kernel documentation provides information on core concepts and configurations, as well as providing :doc:`steps for building Composable Kernel <install/Composable-Kernel-install>`. Some of this information is also available in the `Composable Kernel README file <https://github.com/ROCm/composable_kernel/blob/develop/README.md>`_.

Consult the `AMD Developer Central portal <https://www.amd.com/en/developer.html>`_ for more information about AMD products.

Reporting issues
=================

Use `Github issues <https://github.com/ROCm/composable_kernel/issues>`_
to track public bugs and enhancement requests.

If you encounter an issue with the library, please check if the problem has already been
reported by searching existing issues on GitHub. If your issue seems unique, please submit a new
issue. All reported issues must include:

* A comprehensive description of the problem, including:

  * What did you observe?
  * Why do you think it is a bug (if it seems like one)?
  * What did you expect to happen? What would indicate the resolution of the problem?
  * Are there any known workarounds?

* Your configuration details, including:

  * Which GPU are you using?
  * Which OS version are you on?
  * Which ROCm version are you using?
  * Are you using a Docker image? If so, which one?

* Steps to reproduce the issue, including:

  * What actions trigger the issue? What are the reproduction steps?

    * If you build the library from scratch, what CMake command did you use?

  * How frequently does this issue happen? Does it reproduce every time? Or is it a sporadic issue?

Before submitting any issue, ensure you have addressed all relevant questions from the checklist.

Creating Pull Requests
=======================

You can submit `Pull Requests (PR) on GitHub
<https://github.com/ROCm/composable_kernel/pulls>`_.

All contributors are required to develop their changes on a separate branch and then create a
pull request to merge their changes into the `develop` branch, which is the default
development branch in the Composable Kernel project. All external contributors must use their own
forks of the project to develop their changes.

External Contributor Guidelines
-------------------------------

As an external contributor to this open source project with dozens of active developers, please
follow these essential guidelines to ensure a smooth review and approval process:

**Code Quality and Formatting**

* **Use pre-commit hooks:** Install and use the provided pre-commit hooks that perform clang
  formatting and linting. These can be installed using the ``install_precommit.sh`` script
  located in the ``script/`` folder. This ensures consistent code formatting and catches
  common issues before submission.

* **Keep branches up to date:** Regularly rebase or merge the ``develop`` branch onto your
  feature branch to resolve conflicts properly. This should be done both prior to creating
  your PR and during the review process to maintain compatibility.

**Pull Request Size and Complexity**

* **Maintain manageable PR size:** Keep pull requests to a maximum of approximately 1,000 lines
  of changes to facilitate streamlined review and approval. For larger changes, break them into
  smaller, focused pull requests that can be reviewed independently.

* **Add inline documentation:** Include relevant documentation and comments inline with your
  code changes to help reviewers understand the purpose and implementation details.

**Architectural Changes and Performance**

* **Design documents for major changes:** Major architectural changes must be accompanied by
  comprehensive design documents uploaded with the PR. This helps reviewers understand the
  broader impact and rationale for significant modifications.

* **Performance monitoring:** For changes that may impact build times or runtime performance,
  provide documentation showing before and after performance numbers. This is essential for
  large changes and helps maintain the project's performance standards.

**Review Process for External Contributors**

* **AMD approval for CI:** Forks will require an approver from AMD to trigger continuous
  integration (CI) testing. Please be patient as this approval process is necessary for
  security and resource management.

* **Potential internal collaboration:** Depending on the complexity of your changes, an
  internal AMD developer may need to pull your changes and perform additional fixes or
  modifications prior to merge. This collaborative approach ensures compatibility with
  internal systems and standards.

General Pull Request Requirements
---------------------------------

When submitting a Pull Request you should:

* Describe the change providing information about the motivation for the change and a general
  description of all code modifications.

* Verify and test the change:

  * Run any relevant existing tests.
  * Write new tests if added functionality is not covered by current tests.

* Ensure your changes align with the coding style defined in the ``.clang-format`` file located in
  the project's root directory. We leverage `pre-commit` to run `clang-format` automatically. We
  highly recommend contributors utilize this method to maintain consistent code formatting.
  Instructions on setting up `pre-commit` can be found in the project's
  `README file <https://github.com/ROCm/composable_kernel/blob/develop/README.md>`_

* Link your PR to any related issues:

  * If there is an issue that is resolved by your change, please provide a link to the issue in
    the description of your pull request.

* For larger contributions, structure your change into a sequence of smaller, focused commits, each
  addressing a particular aspect or fix.

Following the above guidelines ensures a seamless review process and faster assistance from our
end.

Thank you for your commitment to enhancing the Composable Kernel project!
