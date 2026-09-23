# Third-Party Notices

The project license does not replace the licenses of bundled dependencies.

- DPLM/ByProt source: `inverse_folding/dplm/LICENSE` (Apache-2.0).
- OpenFold source: `inverse_folding/dplm/vendor/openfold/LICENSE` and its
  `CITATION.cff`. Retain the notices in copied source files.
- Model weights and external tools, including ProteinMPNN, ESMFold2, Protenix,
  AlphaFold3, NetMHCIIpan, and Rosetta, follow their upstream distribution terms.
  They are not included in the source archive.

The bundled DPLM code includes local inverse-folding and sampling changes.
Use the bundled implementation with the checkpoint/config pair documented for
your run; substituting another implementation can change the sampled sequences.
