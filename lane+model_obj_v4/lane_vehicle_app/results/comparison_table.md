| video @start (frames) | LANE % base→v7 | ONE_SIDED % | NONE % | jitter px base→v7 | width jumps | state differs | boxes/frame | dup pairs | flicker gaps | class chg | lane overlay on vehicles base→v7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12.mp4 @0 (150) | 75% → 75% | 25% → 25% | 1% → 1% | 3.13 → 3.13 | 2 → 2 | 0% | 0.13 → 0.14 | 3 → 0 | 1 → 0 | 5 → 1 | 0.000 → 0.000 |
| 12.mp4 @2000 (150) | 97% → 97% | 3% → 3% | 1% → 1% | 1.20 → 1.20 | 0 → 0 | 0% | 0.32 → 0.27 | 5 → 0 | 4 → 0 | 4 → 1 | 0.000 → 0.000 |
| 99.mp4 @0 (150) | 95% → 97% | 4% → 2% | 1% → 1% | 0.64 → 0.63 | 0 → 0 | 2% | 3.67 → 3.81 | 0 → 0 | 10 → 1 | 0 → 0 | 0.000 → 0.000 |
| 99.mp4 @500 (150) | 98% → 98% | 1% → 1% | 1% → 1% | 0.95 → 0.90 | 0 → 0 | 0% | 4.67 → 5.05 | 0 → 2 | 13 → 4 | 0 → 0 | 0.000 → 0.000 |
| 99.mp4 @1500 (150) | 53% → 51% | 8% → 35% | 39% → 13% | 2.69 → 2.72 | 6 → 4 | 47% | 2.85 → 2.58 | 33 → 0 | 20 → 2 | 23 → 2 | 0.033 → 0.000 |
| 99.mp4 @2500 (150) | 63% → 50% | 33% → 43% | 5% → 7% | 1.97 → 2.16 | 0 → 0 | 19% | 2.03 → 2.33 | 4 → 0 | 13 → 9 | 7 → 1 | 0.028 → 0.000 |
| 99.mp4 @3000 (200) | 80% → 80% | 20% → 20% | 1% → 1% | 1.50 → 1.50 | 3 → 3 | 0% | 1.00 → 1.00 | 0 → 0 | 0 → 0 | 0 → 0 | 0.000 → 0.000 |
| 99.mp4 @3560 (150) | 99% → 99% | 0% → 0% | 1% → 1% | 0.88 → 0.82 | 0 → 0 | 0% | 4.56 → 4.10 | 74 → 0 | 32 → 1 | 65 → 1 | 0.000 → 0.000 |
| 99.mp4 @4100 (150) | 97% → 97% | 2% → 3% | 1% → 1% | 0.91 → 1.09 | 0 → 0 | 5% | 3.79 → 3.73 | 25 → 0 | 11 → 3 | 32 → 0 | 0.000 → 0.000 |
| 99.mp4 @5640 (150) | 99% → 99% | 0% → 0% | 1% → 1% | 1.06 → 1.14 | 2 → 1 | 0% | 3.51 → 4.19 | 0 → 0 | 31 → 9 | 3 → 3 | 0.001 → 0.000 |
| nD_20.mp4 @300 (150) | 96% → 96% | 3% → 3% | 1% → 1% | 2.91 → 2.91 | 3 → 3 | 0% | 0.00 → 0.00 | 0 → 0 | 0 → 0 | 0 → 0 | - → - |
| nD_20.mp4 @0 (150) | 98% → 98% | 1% → 1% | 1% → 1% | 0.95 → 0.95 | 0 → 0 | 0% | 0.00 → 0.00 | 0 → 0 | 0 → 0 | 0 → 0 | - → - |

weighted LANE share over all segments: baseline 87%, v7 86% (1850 frames)

zones of displayed vehicles (v7), per segment:
- 12.mp4 @0: {'UNCERTAIN': 11, 'OUTSIDE': 10} | tracker {'created': 5, 'confirmed': 2, 'dup_removed': 0, 'implausible': 0, 'class_switches': 1, 'zone_suppressed': 8}
- 12.mp4 @2000: {'OUTSIDE': 34, 'UNCERTAIN': 5, 'CURRENT': 1} | tracker {'created': 15, 'confirmed': 4, 'dup_removed': 0, 'implausible': 0, 'class_switches': 1, 'zone_suppressed': 14}
- 99.mp4 @0: {'UNCERTAIN': 107, 'RIGHT_ADJ': 194, 'OUTSIDE': 115, 'CURRENT': 153, 'LEFT_ADJ': 3} | tracker {'created': 28, 'confirmed': 14, 'dup_removed': 0, 'implausible': 1, 'class_switches': 0, 'zone_suppressed': 46}
- 99.mp4 @500: {'UNCERTAIN': 412, 'CURRENT': 156, 'RIGHT_ADJ': 134, 'OUTSIDE': 56} | tracker {'created': 16, 'confirmed': 10, 'dup_removed': 0, 'implausible': 0, 'class_switches': 0, 'zone_suppressed': 9}
- 99.mp4 @1500: {'UNCERTAIN': 282, 'CURRENT': 75, 'LEFT_ADJ': 28, 'OUTSIDE': 2} | tracker {'created': 43, 'confirmed': 15, 'dup_removed': 3, 'implausible': 17, 'class_switches': 2, 'zone_suppressed': 10}
- 99.mp4 @2500: {'UNCERTAIN': 216, 'CURRENT': 75, 'LEFT_ADJ': 45, 'OUTSIDE': 14} | tracker {'created': 17, 'confirmed': 10, 'dup_removed': 0, 'implausible': 0, 'class_switches': 1, 'zone_suppressed': 6}
- 99.mp4 @3000: {'UNCERTAIN': 130, 'CURRENT': 25, 'RIGHT_ADJ': 45} | tracker {'created': 1, 'confirmed': 1, 'dup_removed': 0, 'implausible': 0, 'class_switches': 0, 'zone_suppressed': 0}
- 99.mp4 @3560: {'UNCERTAIN': 4, 'RIGHT_ADJ': 296, 'LEFT_ADJ': 149, 'CURRENT': 149, 'OUTSIDE': 17} | tracker {'created': 18, 'confirmed': 6, 'dup_removed': 0, 'implausible': 4, 'class_switches': 1, 'zone_suppressed': 4}
- 99.mp4 @4100: {'UNCERTAIN': 131, 'RIGHT_ADJ': 145, 'LEFT_ADJ': 200, 'OUTSIDE': 83} | tracker {'created': 19, 'confirmed': 10, 'dup_removed': 0, 'implausible': 3, 'class_switches': 0, 'zone_suppressed': 15}
- 99.mp4 @5640: {'UNCERTAIN': 207, 'LEFT_ADJ': 241, 'CURRENT': 13, 'OUTSIDE': 96, 'RIGHT_ADJ': 71} | tracker {'created': 56, 'confirmed': 28, 'dup_removed': 0, 'implausible': 0, 'class_switches': 2, 'zone_suppressed': 65}
- nD_20.mp4 @300: {} | tracker {'created': 0, 'confirmed': 0, 'dup_removed': 0, 'implausible': 0, 'class_switches': 0, 'zone_suppressed': 0}
- nD_20.mp4 @0: {} | tracker {'created': 0, 'confirmed': 0, 'dup_removed': 0, 'implausible': 0, 'class_switches': 0, 'zone_suppressed': 0}
