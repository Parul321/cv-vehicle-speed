# Viva answers (short)
1. **Problem?** Camera-only vehicle speed in km/h. RAFT gives pixel motion; we convert it to metric speed per vehicle.
2. **Novelty vs RAFT repo?** Vehicle masks + tracking, homography calibration, median flow over the road-contact region, km/h output, validation against reference speeds.
3. **Why a homography?** The road is planar, so a 3x3 projective map takes pixel points on the road exactly to ground metres. Needs 4 points with known real positions.
4. **Why the bottom 25% of the mask?** Only road-contact points lie on the ground plane. Roof pixels are above it and give biased distances.
5. **Why the median?** Robust to outlier flow from mask edges, shadows and occlusion.
6. **Formula?** speed(km/h) = ground displacement (m) x fps / frame gap x 3.6.
7. **Why RAFT over Farneback?** Learned all-pairs correlation + iterative updates handle large motion and weak texture; in our run RAFT-large had r=0.92 and MAE 2.2 km/h vs the tracking reference, Farneback r=0.83 and MAE 9.7 (21 vehicles).
8. **RAFT-small vs large?** Small underestimated (bias -5.6 km/h, r 0.54); large fixed it. Cost: slower on CPU.
9. **What is the tracking-only method?** Box foot point through the same homography, distance over time. It is a reference baseline, not ground truth.
10. **Why a ROI / quad check?** The homography is only valid near the calibrated area; far from it errors blow up (we saw 3000+ km/h before restricting).
11. **Error sources?** Wrong calibration distances, camera tilt or shake, mask leakage onto road, motion blur, low resolution, tracker ID switches.
12. **What does frame gap do?** Larger gap -> larger displacement (better SNR) but coarser time resolution.
13. **Validation?** Known-speed passes -> MAE, bias, RMSE; plus cross-method agreement. Be honest: stock clip had no ground truth.
14. **Why no training?** Pretrained YOLO and RAFT already work on road video; contribution is the metric pipeline. Optional: linear speed correction fit on known passes.
15. **Limitations / future work?** Fixed camera, planar road, manual calibration -> auto calibration from lane markings or vanishing points, fine-tune on traffic data, GPU real-time.
16. **How did you check the geometry and formula without real ground truth?** Synthetic scene: a flat vehicle moves at a known speed through a known homography. With the exact flow field our code recovers the speed with 0.00 km/h error. With Farneback it underestimates slightly up to 50 km/h and fails at higher speeds in that setup. A 10% error in the entered road length gives exactly a 10% speed error.
