# Agility Fixture Family

This fixture family was built from a real Agility export supplied locally as `BDC-Mentions - 2026-10-02_09-10.csv`. The source export remains local and is not required for tests.

The CSV fixtures preserve the source Agility column names. Long headline/snippet-like text was shortened where needed to reduce retained copyrighted text while preserving realistic upload and cleaning characteristics. Stable case IDs live in this manifest rather than in production-facing fixture columns.

## Files

| File | Purpose |
| --- | --- |
| `agility_golden_corpus.csv` | Canonical 100-row raw Agility-style input corpus. |
| `agility_malformed_inputs.csv` | Smaller malformed/robustness input corpus. |
| `agility_golden_cleaned_workbook.xlsx` | Cleaned workbook generated through MIG normalization, Basic Cleaning, grouping, and export code using the golden corpus. |
| `agility_golden_multisheet_upload.xlsx` | Multi-sheet upload fixture. Select worksheet `Agility Export` for the intended data. |

## Golden Corpus Coverage

- Includes all source media channels present in the supplied export: ONLINE_NEWS, LINKEDIN, TV, RADIO, FACEBOOK, INSTAGRAM, X, PRINT, YOUTUBE, REDDIT, BLOGS, TIKTOK, PRESS_RELEASE, and BLUESKY.
- Includes exact URL duplicates, syndicated/near-duplicate story families, similar-but-separate stories, unrelated stories, blank optional fields, non-English rows, Unicode/punctuation variation, varied reach and engagement values, zero values, and sparse metadata rows.
- The cleaned workbook was generated from this corpus through the app path so workbook-schema tests can compare against actual MIG output rather than a hand-built approximation.

## Golden Cases

| Case ID | Source row | Media type | Headline / identifying text | Testing reason |
| --- | ---: | --- | --- | --- |
| GC-001 | 5 | ONLINE_NEWS | PurposeMed to Be Acquired by Grindr in US$250 Million Deal | ONLINE_NEWS ordinary positive traditional article with author/outlet/date/reach. |
| GC-002 | 17 | ONLINE_NEWS | Protesters flood the streets to confront Carney | ONLINE_NEWS neutral protest story, unrelated to business funding coverage. |
| GC-003 | 30 | ONLINE_NEWS | Déplacement de Nathalie Drach-Temam, présidente de Sorbonne Université, au Canada du 1er... | ONLINE_NEWS missing language/sentiment/reach fields after upload normalization. |
| GC-004 | 32 | ONLINE_NEWS | 40 M$ pour un nouveau complexe industriel à Saint-Hubert | ONLINE_NEWS French headline with punctuation and missing metrics. |
| GC-005 | 33 | ONLINE_NEWS | $1 million in Alto executive travel claims. | ONLINE_NEWS long source text shortened for fixture, good long-text handling case. |
| GC-006 | 35 | ONLINE_NEWS | Weathering The Tariff Storm: Federal And Provincial Support Programs Available To Canadia... | ONLINE_NEWS support-program tagging contrast. |
| GC-007 | 37 | ONLINE_NEWS | Volatus Aerospace Marks Official Opening of Mirabel Facility as Canadian Manufacturing Op... | ONLINE_NEWS manufacturing/opening story with high reach. |
| GC-008 | 225 | ONLINE_NEWS | Closing the $25,000–$250,000 Gap for Black Entrepreneurs | ONLINE_NEWS Black entrepreneur funding/tagging contrast with very low reach. |
| GC-009 | 228 | ONLINE_NEWS | 10 Resources Helping Black-Owned Businesses Access Contracts, Funding and Support in Canada | ONLINE_NEWS related Black-owned business resource story, similar topic but separate URL/story. |
| GC-010 | 237 | ONLINE_NEWS | Woveo partners with BDC on business microlending program | ONLINE_NEWS BetaKit microlending story for startup/funding topic. |
| GC-011 | 360 | ONLINE_NEWS | Thirty Years of Empowering Young Founders: How Futurpreneur Helps Build Canada's Next Gen... | ONLINE_NEWS small-business/founder story, positive sentiment. |
| GC-012 | 558 | ONLINE_NEWS | Business for Sale in London Ontario: A Complete Guide for Buyers and Sellers | ONLINE_NEWS business-for-sale story, near duplicate cluster with source row 560. |
| GC-013 | 560 | ONLINE_NEWS | Business for Sale in London Ontario: A Complete Guide for Buyers and Sellers | ONLINE_NEWS same business-for-sale story family as source row 558 but different outlet/language. |
| GC-014 | 641 | ONLINE_NEWS | 50% U.S. Tariffs Are Costing Canadian Small Businesses Orders as Trade War Spreads Beyond... | ONLINE_NEWS tariff/small-business story for economic pressure tagging. |
| GC-015 | 1169 | ONLINE_NEWS | The battle to build a global defence bank | Syndicated defence-bank cluster, low-reach outlet. |
| GC-016 | 1170 | ONLINE_NEWS | The battle to build a global defence bank | Syndicated defence-bank cluster, very high-reach outlet. |
| GC-017 | 1196 | ONLINE_NEWS | INSIGHT-The battle to build a global defence bank | Syndicated defence-bank cluster with INSIGHT headline variation. |
| GC-018 | 1218 | ONLINE_NEWS | The battle to build a global defence bank | Syndicated defence-bank cluster from newspaper outlet. |
| GC-019 | 1231 | ONLINE_NEWS | "بنك الدفاع العالمي".. مبادرة تُعيد رسم خريطة التمويل العسكري | Arabic non-English defence-bank story in same broad news family. |
| GC-020 | 3914 | ONLINE_NEWS | At Startupfest, Georges St-Pierre and Harley Finkelstein compare notes on fear, focus, an... | Exact URL duplicate pair with source row 3946, populated metadata. |
| GC-021 | 3946 | ONLINE_NEWS | At Startupfest, Georges St-Pierre and Harley Finkelstein compare notes on fear, focus, an... | Exact URL duplicate pair with source row 3914, sparse metadata/trailing newline headline. |
| GC-022 | 761 | ONLINE_NEWS | Trump says $5,000 dividend checks will 'happen 100 percent' | ONLINE_NEWS unrelated political/economic article for separation coverage. |
| GC-023 | 2 | LINKEDIN | (blank headline) Nigeria in 1min: Economic, Business and Financial Market Headlines – 1... | Selected for channel/date/author/reach diversity. |
| GC-024 | 6 | LINKEDIN | (blank headline) Delighted and proud to share the news that  PurposeMed has been acquir... | Selected for channel/date/author/reach diversity. |
| GC-025 | 8 | LINKEDIN | (blank headline) Check out the latest edition of the OBIO® newsletter, packed with exci... | Selected for channel/date/author/reach diversity. |
| GC-026 | 14 | LINKEDIN | (blank headline) Plus qu’une semaine avant InnovAcet 2026! 🎉  Êtes-vous prêt·e pour une... | Selected for channel/date/author/reach diversity. |
| GC-027 | 18 | LINKEDIN | (blank headline) Blair Health Raises CAD $4.24 Million in Pre-Seed Financing. Toronto-b... | Selected for channel/date/author/reach diversity. |
| GC-028 | 19 | LINKEDIN | (blank headline) To recognize #TruthandReconciliation day, we are celebrating the rich... | Selected for channel/date/author/reach diversity. |
| GC-029 | 26 | LINKEDIN | (blank headline) La réconciliation commence par l’écoute.  Dans cette conversation avec... | Selected for channel/date/author/reach diversity. |
| GC-030 | 27 | LINKEDIN | (blank headline) Reconciliation begins with listening.  In this conversation with Isabe... | Selected for channel/date/author/reach diversity. |
| GC-031 | 101 | LINKEDIN | (blank headline) Plus de 100 000 entrepreneur·es soutenu·es. Une année marquée par l'in... | Selected for channel/date/author/reach diversity. |
| GC-032 | 102 | LINKEDIN | (blank headline) 100,000+ entrepreneurs supported. A year unlike any other. One convers... | Selected for channel/date/author/reach diversity. |
| GC-033 | 425 | LINKEDIN | (blank headline) ✨ Celebrating a Successful Symposium! ✨ The Age of AI: Innovation Meet... | Selected for channel/date/author/reach diversity. |
| GC-034 | 709 | LINKEDIN | (blank headline) 🚀 Fast or Finished? Canadian entrepreneurs continue to prove they're r... | Selected for channel/date/author/reach diversity. |
| GC-035 | 4 | FACEBOOK | (blank headline) ✨ BDO Canada, fier partenaire Or du Prix Femmes d’affaires du Québec 2... | Selected for channel/date/author/reach diversity. |
| GC-036 | 28 | FACEBOOK | (blank headline) I googled and got this... Yes — the core figures in that post are supp... | Selected for channel/date/author/reach diversity. |
| GC-037 | 88 | FACEBOOK | (blank headline) Dans un contexte économique mondial de plus en plus imprévisible, nos... | Selected for channel/date/author/reach diversity. |
| GC-038 | 90 | FACEBOOK | (blank headline) C'est un honneur d'avoir assisté ce matin au petit-déjeuner de la Cham... | Selected for channel/date/author/reach diversity. |
| GC-039 | 1398 | FACEBOOK | (blank headline) Canada is standing up for Canadian workers, businesses, and industries... | Facebook exact duplicate URL/time-proximity pair with source row 1399. |
| GC-040 | 1399 | FACEBOOK | (blank headline) Canada is standing up for Canadian workers, businesses, and industries... | Facebook exact duplicate URL/time-proximity pair with source row 1398. |
| GC-041 | 2528 | FACEBOOK | (blank headline) The Canadian government will lend you the money to buy one. It is call... | Facebook exact duplicate URL/time-proximity pair with source row 2531. |
| GC-042 | 2531 | FACEBOOK | (blank headline) The Canadian government will lend you the money to buy one. It is call... | Facebook exact duplicate URL/time-proximity pair with source row 2528. |
| GC-043 | 24 | INSTAGRAM | (blank headline) La réconciliation commence par l’écoute.  Dans cette conversation avec... | Selected for channel/date/author/reach diversity. |
| GC-044 | 25 | INSTAGRAM | (blank headline) Reconciliation begins with listening.  In this conversation with Isabe... | Selected for channel/date/author/reach diversity. |
| GC-045 | 45 | INSTAGRAM | (blank headline) MEET OUR SPEAKER: BRETT LUCIER SUMMIT 2026 is officially coming — and... | Selected for channel/date/author/reach diversity. |
| GC-046 | 75 | INSTAGRAM | (blank headline) 📸 Retour en images   Ce matin, la #CCMM a eu le plaisir d'accueillir l... | Selected for channel/date/author/reach diversity. |
| GC-047 | 89 | INSTAGRAM | (blank headline) Dans un contexte économique mondial de plus en plus imprévisible, nos... | Selected for channel/date/author/reach diversity. |
| GC-048 | 92 | INSTAGRAM | (blank headline) Small Business Week takes place October 19–25, 2026, and entrepreneurs... | Selected for channel/date/author/reach diversity. |
| GC-049 | 168 | INSTAGRAM | (blank headline) Dave Lohse on Leadership, Growth, and the @bdc_ca Growth Driver Progra... | Selected for channel/date/author/reach diversity. |
| GC-050 | 208 | INSTAGRAM | (blank headline) The right program, funding opportunity or connection can help move you... | Selected for channel/date/author/reach diversity. |
| GC-051 | 7 | X | (blank headline) J'ai littéralement bu ses paroles. Très hâte que vous découvriez mon e... | Selected for channel/date/author/reach diversity. |
| GC-052 | 20 | X | (blank headline) @FoodProfessor The Owner Paul Burke took the Canadian Taxpayers to the... | Selected for channel/date/author/reach diversity. |
| GC-053 | 21 | X | (blank headline) @yourrightmedia1 @WeAreCanProud The Owner Paul Burke took the Canadian... | Selected for channel/date/author/reach diversity. |
| GC-054 | 22 | X | (blank headline) @WeAreCanProud The Owner Paul Burke took the Canadian Taxpayers to the... | Selected for channel/date/author/reach diversity. |
| GC-055 | 29 | X | (blank headline) @Tybernicus17 U.S. has the SBA loan often used for franchises.  Canada... | Selected for channel/date/author/reach diversity. |
| GC-056 | 54 | X | (blank headline) @MarkJCarney Can you make it easier and faster for small/medium sized... | Selected for channel/date/author/reach diversity. |
| GC-057 | 156 | X | (blank headline) @PsudoMike I'd bet the defaults hinge on the underwriting feedback loo... | Selected for channel/date/author/reach diversity. |
| GC-058 | 110 | TV | La Chaîne d'affaires publiques par câble (CPAC-F) | TV French CPAC item, broadcast grouping case. |
| GC-059 | 111 | TV | Cable Public Affairs Channel (CPAC) | TV English CPAC item, same outlet/date as nearby broadcast rows. |
| GC-060 | 112 | TV | Cable Public Affairs Channel (CPAC) | TV English CPAC near-time broadcast duplicate/grouping case. |
| GC-061 | 135 | TV | CTV Toronto | Selected for channel/date/author/reach diversity. |
| GC-062 | 137 | TV | CP24 | Selected for channel/date/author/reach diversity. |
| GC-063 | 147 | TV | Public Record | TV French public record row. |
| GC-064 | 148 | TV | Dossier public | TV French dossier public row, similar but not identical to source row 147. |
| GC-065 | 269 | TV | Littlest Hobo | Selected for channel/date/author/reach diversity. |
| GC-066 | 38 | RADIO | Newstalk 610 CKTB | Selected for channel/date/author/reach diversity. |
| GC-067 | 73 | RADIO | CKAJ 92.5 FM | Selected for channel/date/author/reach diversity. |
| GC-068 | 108 | RADIO | 590 VOCM | Selected for channel/date/author/reach diversity. |
| GC-069 | 183 | RADIO | Newstalk 610 CKTB | Selected for channel/date/author/reach diversity. |
| GC-070 | 308 | RADIO | CBC Radio One CBT (Gander, NL) | Radio negative Newfoundland row with small reach. |
| GC-071 | 309 | RADIO | CBC Radio One CFGB (Happy Valley-Goose Bay, NL) | Radio positive same-market row near source row 310. |
| GC-072 | 310 | RADIO | CBC Radio One 640AM (CBN) | Radio neutral same-market row near source row 309. |
| GC-073 | 472 | RADIO | CJBQ 800 AM | Selected for channel/date/author/reach diversity. |
| GC-074 | 36 | PRINT | Un nouveau siège social | Print French article with blank author and blank URL. |
| GC-075 | 113 | PRINT | Le maire de Campbellton, Michel Soucy, succombe à un arrêt cardiaque | Print mojibake author/unicode normalization case. |
| GC-076 | 142 | PRINT | Going south 'It feels that we're forced to choose sides' | Selected for channel/date/author/reach diversity. |
| GC-077 | 181 | PRINT | L'achat canadien en cinq chiffres | Selected for channel/date/author/reach diversity. |
| GC-078 | 226 | PRINT | Éviter les retardataires en IA | Selected for channel/date/author/reach diversity. |
| GC-079 | 488 | PRINT | Mark Carney wants the U.K. and Germany to join Canada's new defence bank, says lead negot... | Print high-reach English defence-bank story with blank URL. |
| GC-080 | 198 | YOUTUBE | LiqScan Live \| Global Markets — Global Stocks (24 Sep, 21:00 UTC) | Selected for channel/date/author/reach diversity. |
| GC-081 | 206 | YOUTUBE | This means Millions for Canadian Small Businesses | Selected for channel/date/author/reach diversity. |
| GC-082 | 453 | YOUTUBE | LiqScan Live \| Global Markets — Global Stocks (18 Sep, 13:00 UTC) | Selected for channel/date/author/reach diversity. |
| GC-083 | 541 | YOUTUBE | LiqScan Live \| Global Markets — Global Stocks (17 Sep, 03:00 UTC) | Selected for channel/date/author/reach diversity. |
| GC-084 | 1043 | YOUTUBE | LiqScan Live \| Global Markets — Global Stocks (02 Sep, 20:00 UTC) | YouTube exact duplicate URL pair with source row 1054. |
| GC-085 | 1054 | YOUTUBE | LiqScan Live \| Global Markets — Global Stocks (02 Sep, 20:00 UTC) | YouTube exact duplicate URL pair with source row 1043. |
| GC-086 | 3 | REDDIT | Global Business News &amp; Hints— September 30, 2026 — Evening Update — Last 24 Hours (Pa... | Selected for channel/date/author/reach diversity. |
| GC-087 | 64 | REDDIT | Mini 2017 F54 JCW- B48 start up almost stalls on cold start up | Selected for channel/date/author/reach diversity. |
| GC-088 | 155 | REDDIT | Global Business News — September 25, 2026 — Midday Update — Last 12 Hours (Pacific Time) | Selected for channel/date/author/reach diversity. |
| GC-089 | 373 | REDDIT | Global Talent Stream Referral Partners Cut by 29%, 17 Organizations Removed, September 2026 | Selected for channel/date/author/reach diversity. |
| GC-090 | 419 | REDDIT | What Is the Best AI Advisor for Dividend Investing? | Selected for channel/date/author/reach diversity. |
| GC-091 | 42 | BLOGS | Thirty Years of Measuring the Wrong Object | Selected for channel/date/author/reach diversity. |
| GC-092 | 172 | BLOGS | Signs Your Small Business Has Outgrown Its Current Cash Flow System | Selected for channel/date/author/reach diversity. |
| GC-093 | 189 | BLOGS | OMERS Promotes Laura Lenz to Lead Ventures Amid Canada-First Push | Selected for channel/date/author/reach diversity. |
| GC-094 | 790 | TIKTOK | (blank headline) # 🚀 Ready to Start Your Business in Canada? Here's Your Roadmap! Great... | TikTok low-reach English social row. |
| GC-095 | 2504 | TIKTOK | (blank headline) 10 RESSOURCES POUR SE LANCER EN AFFAIRES AU CANADA 🇨🇦 5 business à lan... | TikTok French high-reach social row. |
| GC-096 | 1491 | PRESS_RELEASE | Helcim Raises $53 Million Series C With Participation From Curql Fund | Press release source type retained through media-type normalization. |
| GC-097 | 3481 | PRESS_RELEASE | Government of Canada invests nearly $40.5 million in Waterloo and Brant Region businesses... | Press release with very long government headline shortened for fixture. |
| GC-098 | 4130 | PRESS_RELEASE | Canada targets 10 founding nations for defence bank | Press release sharing a broader defence-bank theme with online cluster. |
| GC-099 | 381 | BLUESKY | (blank headline) Several Canadian defence startups have already made their way through... | Bluesky positive social row. |
| GC-100 | 817 | BLUESKY | (blank headline) Valinor launches tokenized Business Development Company fund on Supers... | Bluesky neutral low-reach social row. |

## Malformed Cases

| Case ID | Source row basis | Testing reason |
| --- | ---: | --- |
| MF-001 | 5 | Malformed date and impossible time; should become an invalid Date warning. |
| MF-002 | 17 | Missing media type; should be detected in upload quality report. |
| MF-003 | 30 | Text in numeric fields; numeric coercion should not crash. |
| MF-004 | 36 | Blank optional URL and author on a print row. |
| MF-005 | 88 | Blank headline and snippet on a social row. |
| MF-006 | 110 | Unicode-heavy headline/snippet that should remain valid text. |
| MF-007 | 198 | Extremely long synthetic headline/snippet without copyrighted article text. |
| MF-008 | 381 | Unexpected media type value THREADS. |
| MF-009 | 790 | Missing published time on a TikTok row. |
| MF-010 | 1169 | Missing SyndicationId in otherwise valid syndicated news row. |
| MF-011 | 1398 | Near duplicate Facebook row with one date/time format. |
| MF-012 | 1399 | Near duplicate Facebook row with ISO date/time format. |
| MF-013 | 1491 | Text in AVE currency field on press release row. |
| MF-014 | 2504 | Blank outlet on high-reach TikTok row. |

## Expected Future Test Uses

- Upload parser and upload-quality warnings: `agility_golden_corpus.csv`, `agility_malformed_inputs.csv`, and `agility_golden_multisheet_upload.xlsx`.
- Basic Cleaning and row reconciliation: `agility_golden_corpus.csv`.
- Duplicate removal and story grouping: exact duplicate pairs GC-020/GC-021, GC-039/GC-040, GC-041/GC-042, and GC-084/GC-085, plus defence-bank cluster GC-015 through GC-019.
- Save/Load and Excel re-entry: `agility_golden_cleaned_workbook.xlsx`.
- Worksheet selection: `agility_golden_multisheet_upload.xlsx`, selecting `Agility Export`.

## Generation Notes

- Source export rows: 4,427. Golden fixture rows: 100. Malformed fixture rows: 14.
- Golden normalized upload warnings: 0.
- Cleaned workbook traditional rows: 49; social rows: 47; duplicate rows: 4; unique story rows: 44.
