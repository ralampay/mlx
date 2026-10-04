# Datasets for zero-shot cross-dataset evaluation

This document describes the **prepared ACDC and MRTMD evaluation datasets**, not
the entirety of either upstream dataset. Counts were checked against the local
annotations and preparation manifests on **2026-10-04**.

Paper assets:

- [BibTeX references](./zero_shot_datasets.bib)
- [LaTeX methods text and tables](./zero_shot_datasets.tex)

Here, *zero-shot* means evaluation on a target dataset without fitting the model,
adapters, normalization statistics, calibration, or hyperparameters on that target.
The six object classes are already known to the detector; this is **cross-dataset
generalization**, not unseen-category or open-vocabulary detection. DAWN is the
adapter-training domain in the [adapter study](./yolox_adapters.md), so it must not
be described as an unseen target for those adapted models.

## Prepared composition

All outputs use the foundation model's class order:
`0 person, 1 bicycle, 2 motorcycle, 3 car, 4 bus, 5 truck`.

| Prepared dataset | Images | Individual object boxes | Crowd regions | Role |
| --- | ---: | ---: | ---: | --- |
| ACDC, full official adverse-condition validation split, mapped to six classes | 406 | 2,685 | 63 | Primary ACDC evaluation, native JSON |
| ACDC, crowd-free subset of that same split | 349 | 2,146 | 0 | Alternative for YOLO-text-only tools |
| MRTMD, deduplicated labelled 1080p release | 3,711 | 69,010 | 0 | Primary MRTMD evaluation |

ACDC's **2,748 mapped annotations** are 2,685 ordinary boxes plus 63 crowd
annotations; crowd regions are not 63 individually enumerated objects. The
349-image subset overlaps the 406-image split and is not an additional dataset.

| Class / model ID | ACDC individuals | ACDC crowds | ACDC crowd-free individuals | MRTMD individuals |
| --- | ---: | ---: | ---: | ---: |
| person / 0 | 567 | 11 | 352 | 21,274 |
| bicycle / 1 | 68 | 22 | 36 | 64 |
| motorcycle / 2 | 59 | 1 | 48 | 15,525 |
| car / 3 | 1,836 | 29 | 1,571 | 30,187 |
| bus / 4 | 49 | 0 | 43 | 337 |
| truck / 5 | 106 | 0 | 96 | 1,623 |
| **Total** | **2,685** | **63** | **2,146** | **69,010** |

## ACDC

### Source and acquisition

[ACDC](https://acdc.vision.ee.ethz.ch/) is the Adverse Conditions Dataset with
Correspondences. Its driving scenes were recorded in Switzerland. The broader
release contains 4,006 adverse-condition images and 4,006 corresponding
normal-condition images. Cite the original ICCV paper for the dataset's
introduction and the extended TPAMI paper for its detection and panoptic tasks:
`sakaridis2021acdc` and `sakaridis2026acdc` in the supplied bibliography.
See the [original paper](https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html),
[extended paper](https://doi.org/10.1109/TPAMI.2025.3633063), and
[capture provenance](https://acdc.vision.ee.ethz.ch/dataprotection).

The local preparation includes **only the 406 adverse-condition validation
images**, at 1920 × 1080. It does not include the normal-condition reference
images or the training/test images. The official detection ZIP also contains
training annotations; retaining those files does not imply their images were
downloaded. Official test ground truth is withheld.

| Condition | Full validation | Crowd-free subset |
| --- | ---: | ---: |
| Fog | 100 | 97 |
| Night | 106 | 81 |
| Rain | 100 | 84 |
| Snow | 100 | 87 |
| **Total** | **406** | **349** |

Acquisition record:

- Detection annotations: official `gt_detection_trainval.zip` from the
  [ACDC download service](https://acdc.vision.ee.ethz.ch/download).
  Publisher MD5 verified: `32598aacfe0f3c5138262849be8f35f3`.
- Images: selected ZIP entries from the
  [pcdat1003/ACDC mirror](https://huggingface.co/datasets/pcdat1003/ACDC/tree/e09c38d6e347cd7e8ada42413851d610b311f891),
  revision `e09c38d6e347cd7e8ada42413851d610b311f891`.
- The first 64 MiB of the mirror archive matched the official archive byte for
  byte. Each extracted image passed ZIP CRC32, byte-count, PNG-integrity, and
  annotation-dimension checks. **The full image archive was not downloaded or
  verified against its publisher MD5.**
- The publisher's non-commercial license is retained at
  `acdc/original/License.pdf`; consult it before redistribution or other use.

### Taxonomy and preparation

| ACDC source ID / class | Prepared ID / class | Policy |
| --- | --- | --- |
| 24 person | 0 person | Retain |
| 25 rider | 0 person | Merge the human rider into person |
| 33 bicycle | 1 bicycle | Retain vehicle box separately |
| 32 motorcycle | 2 motorcycle | Retain vehicle box separately |
| 26 car | 3 car | Retain |
| 28 bus | 4 bus | Retain |
| 27 truck | 5 truck | Retain |
| 31 train | — | Remove 46 annotations; retain their images |

The original validation JSON contains 2,794 annotations. Removing the 46 train
annotations leaves 2,748. Source boxes, segmentation-derived areas, segmentation
data, and crowd flags are retained. Annotation IDs are reassigned to positive
unique values; original IDs remain in `source_annotation_id`.

Use native JSON for the 406-image evaluation so that the evaluator can preserve
crowd semantics. The alternative YOLO-text export removes **all 57 images
containing crowd annotations**, rather than silently dropping crowd boxes from
otherwise retained images. This changes the scene distribution toward less
crowded images. Scores on the two versions must be identified separately.

This six-class mapping is a custom evaluation protocol, not ACDC's official
eight-class detection benchmark. Its scores are not directly comparable to the
official leaderboard. Mapping rider to person also changes source semantics and
must be disclosed.

## MRTMD

### Source and acquisition

[MRTMD](https://github.com/OVD-Labs/MRTMD) is the Multi-Resolution Traffic
Monitoring Dataset. Its authors describe curated Creative Commons traffic videos
from YouTube, covering several camera perspectives and provided at multiple
resolutions. Cite `bugeja2025mrtmd`; see the
[published paper](https://doi.org/10.1109/ACCESS.2025.3585986).

The local copy uses the **labelled 1080p release only**, at 1920 × 1080, pinned to
repository commit `664f6786e884e41cc95fd2a36949833223acea64`. Images were fetched
directly from that repository, with each file checked against its Git blob SHA1,
byte count, JPEG metadata, and annotation dimensions. SHA256 values are recorded
locally. Other resolutions and number-plate crops were not included.

The pinned JSON has 3,733 image records and 69,348 annotations. After removing
22 duplicate records referring to the same filenames with identical annotations,
the prepared set has **3,711 images and 69,010 boxes**. These are release-specific
counts derived from the downloaded files; do not substitute headline counts from
the paper or repository description. There was no new random train/test split:
the complete deduplicated labelled 1080p release is reserved for evaluation.

| Source video identifier | Prepared images |
| --- | ---: |
| `video1` | 919 |
| `video4` | 387 |
| `video5` | 2,405 |
| **Total** | **3,711** |

The repository retains an MIT license in `mrtmd/original/LICENSE`; the original
annotation JSON declares CC BY 4.0, and the paper describes Creative Commons
source videos. Preserve these separate notices rather than assuming one license
statement replaces all source terms.

### Taxonomy and preparation

| MRTMD source ID / class | Prepared ID / class |
| --- | --- |
| 1 person | 0 person |
| 2 bicycle | 1 bicycle |
| 4 motorcycle | 2 motorcycle |
| 3 car | 3 car |
| 6 bus | 4 bus |
| 8 truck | 5 truck |

No class is merged or removed. Both native JSON and YOLO-text exports represent
the same images and boxes, and the source contains no crowd annotations.
Preparation makes these additional corrections:

1. Remove the 22 redundant image records and their 338 repeated annotations.
2. Clip 575 retained boxes at image boundaries. The maximum source overflow is
   0.25 pixels; the original annotations remain unchanged on disk.
3. Recompute `area` from the final 1080p box. All source areas are approximately
   four times their 1080p box areas, consistent with an unscaled 2160p area field.
   Original areas remain in `source_area`. These annotations have no segmentation
   polygons, so box area is used.
4. Assign positive unique annotation IDs and preserve `source_annotation_id`.

The same source frames appear at other resolutions upstream. Do not combine those
variants and count them as independent samples. The 3,711 images come from only
three videos; frame-level sample size overstates independent scene diversity.
Report per-video performance where possible. With only three groups, even
video-level confidence intervals will be unstable. Bicycle AP is particularly
uncertain because there are only 64 bicycle annotations.

## Provenance and evaluation boundaries

The cached foundation manifest lists `bdd100k`, `coco`, `mot17`, `mot20`, `pcb`,
and `visdrone`. It does not list ImageNet as an image source; this does not rule
out ImageNet or other data having been used in upstream backbone pretraining.
ACDC and MRTMD were selected from their documented provenance rather than from
subsets of those foundation datasets. Their use of the **COCO annotation format**
does not mean their images originate in COCO.

SHA256 comparisons found no exact matches between ACDC and the foundation/DAWN,
or between MRTMD and the foundation/DAWN/ACDC. These checks establish absence of
**exact file matches in the checked local artifacts**, not absence of all
training contamination, re-encoded copies, shared source videos, or perceptual
near duplicates. Do not claim universal leakage-free independence in the paper.

For a zero-shot comparison, fix confidence/NMS thresholds, image size, preprocessing,
checkpoint selection, and all adaptation settings before inspecting target-set
performance. Avoid target-specific BN updates or test-time adaptation unless
reported as a separate protocol. Report each dataset separately; pooling the
images would heavily weight MRTMD and its dominant video. State the evaluator,
IoU thresholds, maximum detections, treatment of crowds, and class mapping when
reporting AP. No accuracy measurements were produced during dataset preparation.

## Local artifacts and reproduction

The dataset root is `~/Desktop/datasets/object-detection`. Move each dataset's
entire directory when relocating it: `processed/images/val` uses relative
symlinks to `original`, while processed labels are separate files.

| Dataset | Primary configuration | Alternative configuration |
| --- | --- | --- |
| ACDC | `acdc/processed/data.yaml` | `acdc/processed/yolo_crowdfree.yaml` |
| MRTMD | `mrtmd/processed/data.yaml` | `mrtmd/processed/yolo.yaml` |

Each YAML aliases `val` and `test` to the **same** evaluation data and defines no
training split. `test` is a convenience alias, not the upstream official test set.
Each dataset directory contains `manifest.json`, `validation.json`, source
annotations, a provenance README, and the preparation script (`prepare_validation.py`
for ACDC; `prepare.py` for MRTMD). MRTMD also includes `validate.py`. The scripts
use the saved source metadata/index files; they are not fresh standalone installers.

For a paper artifact, retain the manifests, pinned source versions, preparation
scripts, mapping policies, and run configuration alongside results. Image hashes
are in the manifests. Both dataset loaders were checked without loading a model
or using a GPU; individual label coordinates were also compared between MRTMD's
YOLO and JSON exports.

## LaTeX usage and citation verification

Copy the `.tex` and `.bib` files into the paper project. With classic BibTeX:

```latex
% Preamble
\usepackage{booktabs}
\usepackage{url}

% Body: adjust the path if the fragment is in a subdirectory.
\input{zero_shot_datasets}

% Use the bibliography style required by your venue.
\bibliographystyle{IEEEtran}
\bibliography{zero_shot_datasets}
```

The fragment describes prepared data and a proposed evaluation protocol; check
its claims against the actual experiment before submission. It contains no
invented accuracy results. Cite the dataset authors for the datasets and describe
our mappings and corrections separately as local preparation.

Bibliography metadata were checked against the
[CVF ICCV record](https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html),
[publisher-deposited ACDC journal record](https://api.crossref.org/works/10.1109/TPAMI.2025.3633063),
and [publisher-deposited MRTMD record](https://api.crossref.org/works/10.1109/ACCESS.2025.3585986).
The extended ACDC article has a 2025 DOI/early publication history but its final
issue is **48(3), 2970–2988, March 2026**; the BibTeX uses that final issue year.
The optional ICCV entry uses CVF's published BibTeX and pagination
(10765–10775), rather than mixing it with the different IEEE pagination.
