# Fix-loop declaration (written and committed before any fix code)

Baseline: tag `fixloop-before` (pipeline `0.1.0+e54835b`), report `reports/synthetic/benchmark.md` (benchmark-v1).

## 1. Worst-performing gate, with the failing number

**Photo-tier whole-property stitch** — fails on **2/2** multi-room photo captures:

| Capture | Adjacency | Footprint error (gate +-8 %) | GT in ci90 |
|---|---|---|---|
| home_multiroom_photo_t1 | **wrong** (3 doorways missing, 2 invented) | **+28.2 %** | no |
| holdout2026_multiroom_photo_t1 | **wrong** (north1-north2 missing) | -4.7 % | yes |

It is the worst gate by distance: a hard gate ("a photo path that handles single rooms only fails this row") failing on
every capture, with a footprint error 3.5x the tolerance. (Next: LiDAR opening widths, 51 % vs 85 %.)

## 2. Root-cause hypotheses and evidence

Feedback loop: `python fixloop/probe_photo_stitch.py` (deterministic, ~1 min) prints per-room area error, doorways
and solver pairs for both captures.

**Defect A (home): the hall leaks into the bath through a gap in header-band evidence.**
* The probe shows the photo-tier hall as a 10-wall, 20.53 m² polygon (GT 6.82 m², 4 walls); every other room is
  within 2-14 % of GT. The +13.7 m² excess is the whole footprint error, and it also breaks room matching and
  therefore adjacency.
* Debug top view (`scan/render/debug.py`) of the hall folder: the polygon is hall + bath, joined across the hall's
  south wall.
* Rooms merge across a cell edge unless the wall line shows header-band evidence (top 0.3 m below the ceiling) on
  >= 40 % of that edge. Printing the south-wall line's header coverage over the bath's x-range gives
  `#......######...` - a 0.3 m gap next to the bath doorway (the upper wall there is only seen at grazing angles
  from a 1.1 m wide hall). The arrangement has a column cell narrower than that gap, so that single edge has no
  header evidence and the bath's floor (seen through the doorway) joins the hall.
* Control: the same capture reconstructed with a different (equally plausible) oracle error draw gives a correct
  7.41 m² hall - the failure is fragility to small unobserved patches, not a systematic offset.

**Defect B (holdout): a doorway between two near-identical rooms is taken for a mirror.**
* The holdout has no mirror, yet the run warns `mirror detected ... (~0.92 x 1.76 m)`, and north1's second doorway
  (to north2) is missing, so the north1-north2 adjacency cannot be found.
* Disabling the mirror filter for an experiment (monkeypatch) makes the holdout capture pass outright: adjacency
  correct (5/5 doorways), footprint -4.7 %, GT in ci90.
* Mechanism: the mirror test reflects wall-parallel surfaces seen through the wall across the wall plane. north2's
  far wall is 3.37 m behind the shared wall; north1 is 3.39 m wide, so the reflection lands 2 cm from north1's own
  far wall and "matches" within the photo tolerance (2 % of the folded distance). A real doorway is told apart by
  its **jamb reveals**: surfaces perpendicular to the wall inside the wall thickness, seen from this side. A mirror
  is flush with the wall and has none.

## 3. The fix and the predicted numbers

* A: close gaps of up to 0.6 m in each wall line's header (and occupancy) coverage before the per-edge separation
  test (unobserved patches above a doorway are bridged; open-plan archways are wider than 0.6 m).
* B: a see-through blob with jamb-reveal evidence on both sides is an opening, never a mirror.

Predicted after the fix:

| Capture | Adjacency | Footprint | GT in ci90 | Gate |
|---|---|---|---|---|
| home_multiroom_photo_t1 | correct | within +-5 % (hall ~7.4 m²) | yes | PASS |
| holdout2026_multiroom_photo_t1 | correct | -4.7 % (unchanged) | yes | PASS |

No other gate should move by more than noise (LiDAR and video adjacency stay correct); this is checked by rerunning the
whole benchmark, not only this gate.
