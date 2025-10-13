Installation
============

Get the Syna source

.. code-block:: bash

   git clone https://github.com/sql-hkr/syna.git
   cd syna
   uv venv
   source .venv/bin/activate
   uv sync

Or, from `PyPI <https://pypi.org/project/syna/>`_:

.. code-block:: bash

   uv add syna

.. important::

   To visualize the computation graph, you need to install `Graphviz <https://graphviz.org>`_.

   .. code-block:: bash

      brew install graphviz # macOS
      sudo apt install graphviz # Ubuntu