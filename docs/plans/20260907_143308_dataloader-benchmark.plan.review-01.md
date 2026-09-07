# Review: сравнение ридеров в PyTorch DataLoader

| Field | Value |
|---|---|
| Artifact | `review` |
| Reviewed artifact | `/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.plan.md` |
| Reviewed artifact type | `plan` |
| Reviewed artifact ID | `20260907_143308_dataloader-benchmark` |
| Reviewed revision | `1` |
| Reviewed artifact SHA-256 | `91790b2057f4651882f7b550b98d5b72528c4154a8fb95666ce9e97a02d3994d` |
| Bound artifact SHA-256 | `43d443144364631a33c506975ff1ace576b858e2b4674f2d9a397e73f29499f6` |
| Review round | `01` |
| Reviewer | `independent-agent` |
| Review context | `fresh` |
| Reviewer profile | `work_artifact_reviewer` |
| Model / reasoning | `gpt-6-astra / xhigh` |
| Agent/thread ID | `/root/dataloader_plan_review_01` |
| Input fingerprint | `6574eaa12640d09888b66e989f24dbc8a914523611eebe7830251cb5c2bb85ab` |
| Input fingerprint definition | SHA-256 точных UTF-8 байтов входного манифеста, включая завершающий LF |
| Created | `2026-09-07T17:00:35+04:00` |
| Verdict | `approved` |

## Summary

План согласован с принятой задачей ревизии 2. Он определяет четыре самостоятельных пути декодирования, общий Tensor-transform, независимый от жизненного цикла workers порядок выборки, границы времени, изоляцию конфигураций и наблюдение CPU/RSS с явной неполнотой.

Файлы имеют определённые обязанности; зависимости задач последовательны, пересекающееся владение обозначено. Проверки включают независимые ожидаемые результаты, реальные DataLoader-проходы и ошибки процессов. Приёмочный запуск ограничен фиксированной матрицей и связан с одобренным чистым коммитом. Существенных решений, неявно переложенных на исполнителя, не выявлено.

Отпечаток манифеста и хэши всех 36 файлов совпали. План, исходная задача, её approval reference, рубрика и оба применимых AGENTS.md прочитаны полностью. Применены Common rubric и Plan rubric. Проверка оставалась только чтением.

## Findings

No findings.

## Traceability checks

| Source requirements | Acceptance criteria | Implementation and verification evidence |
|---|---|---|
| R1, R6 | AC1 | Tasks 1–3: неизменяемый snapshot, сохранение порядка и повторов, отдельный EpochSampler. Task 2 проверяет фактически полученные индексы между ридерами, workers, persistence и независимыми процессами; V1/V2/V6. |
| R2, R3, C3 | AC2 | Tasks 1–2: четыре точных decode paths, header validation, безопасная общая граница ndarray/Tensor, RGB/uint8 и отсутствие scaling. Проверяются identity входа, strides, мутация и lifetime; V1/V2. |
| R4, C4 | AC3 | Task 2: фиксированные resize/crop, численные размеры и crop anchors, независимые координатные и аналитические эталоны. Native-size collation и ошибка несовместимых размеров заданы; V2. |
| R5, R6, R9 | AC4 | Tasks 1–3 и 5: CLI/API-конфиг, requested/effective prefetch, недопустимые сочетания, фактические счётчики 4/4/2 и 4/4, zero-output отказ; V1–V3. |
| R7, R8, C4 | AC5 | Task 3: таймер непосредственно перед `iter(loader)` и после exhaustion, подготовка и отчёт вне окна, один loader, first/subsequent и отсутствие скрытой эпохи. Fake-clock и реальные lifecycle tests; V2. |
| R9, R11 | AC6 | Tasks 1, 3, 5: формулы по фактическим изображениям, failed-epoch исключение, поздняя медиана, согласованные API/table/export и JSON round-trip; V1–V3/V6. |
| R10, C3 | AC7 | Tasks 1–3: unsupported исходники без повторной фильтрации, corrupt/None/worker failures, stat change, reader/path/stage diagnostics; V1/V2. |
| R11, C1, C2, C5 | AC8 | Tasks 3–6: реальные workers 0/2 через spawn, pinning requested/observed, cleanup после успеха/ошибки/отмены, base-only окружение; V2–V5. |
| R2, R5, R11, C1, C5 | AC9 | Tasks 5–6: документация установки и повторных запусков, единицы и ограничения, legacy CLI regressions, все Python-команды через `uv`; V3–V5. |
| R12, R13, C6 | AC10 | Task 4: внешний sampler, правильная группа процессов, собственные CPU counters, weighted mean 175%, max 200%, RSS peak суммы 150 MiB. Real monitoring и off без psutil; V1/V2/V6. |
| R12, R13, C6 | AC11 | Task 4: PID reuse, исчезновение, AccessDenied, отсутствующие counters, короткие окна, cross-boundary exclusion, отдельные coverage/completeness и sampler failure; V1/V2. |
| R7, R11, R12, C7 | AC12 | Tasks 3–4 и 6: fresh subprocess, baseline после подготовки, samples вне consumer, четыре вызова A/B/B/A и проверка независимых identities; V2/V6. |

Все R1–R13, C1–C7 и AC1–AC12 имеют реализацию и содержательную проверку. Проверки скорости не подменяют проверку корректности.

Дополнительно подтверждены следующие основания плана:

- **Принятая задача.** Указаны точные revision 2 и approval reference. Текущий SHA-256 задачи `c4c316e2538ef2759418bf04fb238656dd02c3d51f2ffec7f80f61d7bc1dfd33` совпадает с Bound artifact SHA-256 её ревью. Исторические утверждения о предыдущих преобразованиях метаданных не использовались вместо этой проверки.
- **Planning base.** Canonical repository разрешается в `/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark`; HEAD равен `c0f450674620cc8119a73fc4222daf3bc84eb7e3`, ветка — `aya/dataloader-benchmark`. Git status пуст. Хэши всех 20 перечисленных файлов canonical repository совпадают с соответствующими файлами базового коммита.
- **Drift.** Разрешённый `benchmark.py` действительно не содержит описанных в исходной задаче незакоммиченных shuffle/byte-warmup изменений. План явно учитывает это различие и предусматривает отдельные регрессии при последующем объединении.
- **Совместимость CLI.** Существующие `_KNOWN_COMMANDS`, `_BENCHMARK_FLAG_ALLOWLIST`, ленивые обработчики и тест `test_benchmark_nw_zero_means_all_cpus` согласуются с предложенным добавлением подкоманды без смены старой семантики.
- **Версии и геометрия.** Базовый lock содержит torch 2.10.0, torchvision 0.25.0 и OpenCV 4.13.0.90. Локальный METADATA torchvision требует torch 2.10.0. Разрешённые исходники подтверждают `decode_image` с RGB/EXIF-параметрами, формулу resize, ties-to-even crop anchors и сохранение выходного uint8 при возможном внутреннем float-пути.
- **Lifecycle.** Локальный DataLoader подтверждает различие persistent/non-persistent итераторов, отдельное использование generator, ограничения параметров и приватный teardown. План ограничивает обращение к нему одной функцией, фиксирует версию и требует проверки внешнего process-group cleanup.
- **Source validation.** Локальные Pillow plugins подтверждают необходимость проверки исходной глубины PNG отдельно от отображаемого RGB mode. План включает IHDR/SOF и отказ до успешного decoder path.
- **Containment.** Существующие runnable roots и пути file map канонизируются внутри worktree. В проверенных цепочках существующих предков symlink escape не обнаружен. Оба record root через существующего предка `/home/aya/.codex` разрешаются вне обоих checkout. Их создание, permissions и повторные проверки будущих файлов явно назначены handoff/implementation review.
- **Run contract.** Зафиксированы S0→S1→S2→S3, 16 конфигураций плюс A/B/B/A, команды, версии, детерминированные данные, бюджеты, headroom, единственная активная попытка, artifacts и структурное правило успеха. Определены прекращение при ошибке, cancellation, отсутствие автоматического retry/resume и обязательный новый SHA/review после исправления кода.
- **Operations и rollback.** Есть versioned local export, ошибки стадий, permissions, запрет overwrite и внешней публикации, ограничения retention `/tmp`, optional rollout и откат без миграции изображений. Workflow/control records отделены от исполняемого дерева.

## Residual risks and minor notes

- Это ревью плана. Тесты, benchmark workloads, установка зависимостей и исполнение Python не выполнялись. Практическая квалификация Python 3.12/3.13, четырёх backend paths и cleanup остаётся обязательной работой V1–V7.
- OpenCV native implementation и psutil package не входили в разрешённый набор исходников. Их фактическая доступность и поведение должны подтверждаться запланированными preflight и интеграционными проверками; сеть не использовалась.
- Новые product/test-файлы и оба record root ещё не созданы. Их окончательный containment, permissions и import origin сейчас не доказаны; план требует повторить проверки после создания.
- Существование копий workflow inputs в worktree проверено метаданными. Их содержимое не читалось вне закрытого списка; побайтное совпадение при принятии и handoff остаётся назначенной проверкой.
- Историческая корректность прежних metadata updates исходной задачи не реконструировалась. Подтверждена текущая точная привязка task revision 2 к approval reference.
- Приватный torch teardown, OS cache, shared RSS pages, неатомарные обходы и пропущенные короткие процессы остаются рисками. План явно ограничивает соответствующие гарантии и задаёт проверки ошибок и неполноты.
- Одобрение плана не удостоверяет будущую реализацию и не заменяет обязательный implementation review чистого коммита.

## Verdict rationale

`approved`: по Common rubric и Plan rubric отсутствуют blocker/major. План самодостаточен относительно явно указанных принятых источников, сохраняет область задачи и содержит проверяемые интерфейсы, ответственность файлов и выполнимый порядок работ.

Выбранные технические решения согласуются с доступными исходниками. Неопределённость будущего исполнения обозначена как проверка квалификации, а не как основание считать AC выполненными. Финальная приёмка проверяет требуемый результат через реальный публичный путь и ограниченный run contract.

## Correction handoff

- **Artifact revision reviewed:** `1`.
- **Reviewed artifact SHA-256:** `91790b2057f4651882f7b550b98d5b72528c4154a8fb95666ce9e97a02d3994d`.
- **Bound artifact SHA-256:** `43d443144364631a33c506975ff1ace576b858e2b4674f2d9a397e73f29499f6`.
- **Binding state:** `complete` только когда текущий SHA-256 артефакта равен указанному Bound artifact SHA-256; иначе `incomplete`.
- **Created:** `2026-09-07T17:00:35+04:00`.
- **Final sidecar path:** `/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.plan.review-01.md`.
- **Blocking finding IDs:** отсутствуют.
- **Suggested next action:** содержательные исправления по этому ревью не требуются. После публикации review binding и принятия плана допустим implementation handoff в новую session с указанным canonical worktree. V6 допускается после отдельного одобрения точного коммита реализации.

## Манифест входного набора (метаданные записи)

<details>
<summary>Зафиксированные файлы независимого ревью</summary>

```json
{
  "artifact_revision": 1,
  "assignment": "20260907_143308_dataloader-benchmark.plan.review-01",
  "canonical_repository": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark",
  "files": [
    {
      "path": "/home/aya/.codex/plugins/cache/albu-workflows/codex-workflows/1.0.0-dev.0+codex.20260907085239/skills/review-work-artifact/references/review-artifact.md",
      "sha256": "57fff0cc9ecc14eabedb3687eefd8698468a1085d83f4d87e524cae39defbce3"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/PIL/Image.py",
      "sha256": "3e5ecdcc3e8800749901e61c8108bd3c49063d3ba444fa0b62b0c18af022007f"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/PIL/JpegImagePlugin.py",
      "sha256": "9162d130d4d5adf707688a59a5c2ec68bb7169e7eb7a759d20b656c197d8119f"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/PIL/PngImagePlugin.py",
      "sha256": "5da5590ca2487f7ec2a7db5266bef89185854d57d5e4f619f507a8f2955b7aaa"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torch/utils/data/_utils/worker.py",
      "sha256": "c563230455372022ceef8c196b28dd892e753cd7f21a76311dac20bb844938eb"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torch/utils/data/dataloader.py",
      "sha256": "cb8d82b0de999f21349b7dd261d30919fcf692f0b1d7504ca2594e88e8fd0e96"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torch/utils/data/sampler.py",
      "sha256": "817f818a4897f0c12569aa88edb951c3d9912cc20b4a47d62bff3b70cadf1ac5"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torchvision-0.25.0.dist-info/METADATA",
      "sha256": "b13b6c09eb07623dd666e6eebd6e4af1c9f5c922cbe8be54802aba7931df4ba4"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torchvision/io/image.py",
      "sha256": "6880cf9408bcdc81a6d2655ee96c062875bb0d6367df9825a98d6c436fe9deed"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torchvision/transforms/functional.py",
      "sha256": "0b2fdf8e6764f4be0113acd95b58d3b14158c13fa2374ed93fbbe1f4b8730ded"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torchvision/transforms/v2/functional/_geometry.py",
      "sha256": "1e4f4d2ff22a6d0b7f2b8a09f28e57a327ed8f067c6f17e5d1b359af6b440325"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.venv/lib/python3.13/site-packages/torchvision/transforms/v2/functional/_type_conversion.py",
      "sha256": "efcc25d1d34fc17d3c8ce096e8a719486cbc44042acb132d76b4d44154255133"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/.gitignore",
      "sha256": "7aa087b7281ec44485f0c1a68c41c23cf11a7dff32572716211eae1463f6ba80"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/AGENTS.md",
      "sha256": "ce3e9ada50dd3bef35344136bb3f963e195558ff636bff05874e2674de6e784f"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/Makefile",
      "sha256": "5ed615d50ae1c560f790e1a73d59a4782aa46189b922925ae6e7df209eca4535"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/README.md",
      "sha256": "430c7003ce0b9733b2199fb9024825d71b04af96ea74e6ada5ad4495d9dc907a"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/pyproject.toml",
      "sha256": "01a7ba3542c89a59ba84c809d9a3106c248b487c77a59fe2c7e41baa7e720e93"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/src/imgread_benchmark/__init__.py",
      "sha256": "386f04c1cb3a20ea0076e26c6b9618fd509b56eae33a3c70bec1c0311a92ee0f"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/src/imgread_benchmark/argparse_compat.py",
      "sha256": "52b48f901bab749c4ef43188dff058e00765bd8d3c7e67c04d408c726a75b07a"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/src/imgread_benchmark/benchmark.py",
      "sha256": "8e72f9cf26916db14aedeb85c0edebaf0d9a43a5c711578371da2e5ec47b8984"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/src/imgread_benchmark/cli.py",
      "sha256": "533fc51bce800a21a984d43b35d2df3e7f4bc38cc65224dde18b9868ee1ce487"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/src/imgread_benchmark/get_img_filenames.py",
      "sha256": "3c140523a793c573dd6c87bfecd5a2a7d656bae7da4eea03c752584b6038ccac"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/src/imgread_benchmark/img_libs/PIL.py",
      "sha256": "3cc0f7e02e7b0ec88269193f5dce2539496eccc01632b055c412587b3cacda70"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/src/imgread_benchmark/img_libs/cv2.py",
      "sha256": "0dd342f13cd289c3afa00eff695d7dec1cbe6a4908a043e32902680b2d206ca5"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/src/imgread_benchmark/img_libs/torchvision.py",
      "sha256": "a9e4a2447e5ecdd3d31d6bcbae43d43fde8fb1e07ca157ce365fb25f9c1fb5f2"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/src/imgread_benchmark/read_img.py",
      "sha256": "71657ea4ab3cd195f257dd59391c63e2cb071214b321c8d4d51a2e14bb345729"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/tests/test_benchmark_lazy.py",
      "sha256": "286dbb81c40c12772dbc7ff80ff1cb3175911f90d6f3f1328c85fffe9fcf66c3"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/tests/test_cli_unified.py",
      "sha256": "15be8b7a06896e597848c8d7f7c0c8b084a45ef7c722b955a67838c2926d8912"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/tests/test_get_img_filenames.py",
      "sha256": "9bdb16629282bddbd91c5b747daee80ca86cb029a2ea03eee6271f0a3f70e9c5"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/tests/test_image_libs.py",
      "sha256": "9489d5762378ec368c02ab13b1edd844e7c6ae0405d78c1827553600de2ed91e"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/tests/test_lazy_imports.py",
      "sha256": "91c8cd087fcacee0f90a21bb7decb4a4bf213a3e2351d0988259e3585cf3bad6"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/.worktrees/dataloader-benchmark/uv.lock",
      "sha256": "846e73d08120ec2483d26f05ffb004f9096c036b6fce6d3e414b351c2b4f4244"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/AGENTS.md",
      "sha256": "ce3e9ada50dd3bef35344136bb3f963e195558ff636bff05874e2674de6e784f"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.plan.md",
      "sha256": "91790b2057f4651882f7b550b98d5b72528c4154a8fb95666ce9e97a02d3994d"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.task.md",
      "sha256": "c4c316e2538ef2759418bf04fb238656dd02c3d51f2ffec7f80f61d7bc1dfd33"
    },
    {
      "path": "/home/aya/Prj/imgread_benchmark/docs/plans/20260907_143308_dataloader-benchmark.task.review-01.md",
      "sha256": "71a07f6880800b27f81310e4abec148a746de676d8eb4d189476ccb7aa49ebff"
    }
  ],
  "planning_base_sha": "c0f450674620cc8119a73fc4222daf3bc84eb7e3"
}
```

</details>
