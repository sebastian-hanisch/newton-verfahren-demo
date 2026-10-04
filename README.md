# Newton-Verfahren – Krümmung nutzen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-newton-verfahren-demo.streamlit.app/)**

Stück 2 der **Nichtlineare-Optimierung-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Gradientenabstieg (Stück 1) nutzt nur den Gradienten. Das **Newton-Verfahren** nutzt zusätzlich
die **Krümmung** (Hesse-Matrix) und konvergiert lokal **quadratisch** statt nur linear –
unabhängig von der Konditionszahl. Der Preis: ohne Sicherung kann der volle Schritt fern vom
Optimum divergieren. Diese Demo zeigt beides – den starken lokalen Vorteil und die reale
Schwäche – gemessen statt behauptet.

**Einordnung in die Reihe:**

```
Gradientenabstieg (WURZEL)                       [gebaut]
 └─ Newton-Verfahren                             [DIESES STÜCK]
      └─ Quasi-Newton (BFGS/L-BFGS)               [gebaut]
           └─ Lagrange-Multiplikatoren/KKT        [gebaut]
                ├─ Straf-/Barriere-Verfahren      [gebaut]
                └─ SQP                            [gebaut]
                     └─ Innere-Punkte-Verfahren   [gebaut]
 └─ Stochastische Gradientenverfahren             [gebaut, letztes Stück]
```

**Ergebnis in Kürze:** Newton löst eine Quadratik $f(x)=\tfrac12x^\top Ax$ in **genau einem
Schritt** – unabhängig von der Konditionszahl κ (bei κ=200 braucht Gradientenabstieg mit
Backtracking rund 1600 bis 1900 Schritte für dieselbe Genauigkeit, gemessen mit der gradient-descent-demo
auf 10 Zufallsquadratiken mit d=5 bzw. d=10; bei κ=100 sind es dort 887). Nahe am Optimum verdoppelt sich
die Zahl der korrekten Nachkommastellen mit jedem Schritt (quadratische Konvergenz, gemessener
Exponent ≈1,85–1,98). **Echte Plan-Korrektur:** die ursprüngliche Erwartung ("Newton scheitert
bei manchen Startpunkten fern vom Optimum auf der Rosenbrock-Funktion") wurde **widerlegt** – bei
34 getesteten Startpunkten (30 Zufallspunkte in [−2, 2]², dazu die klassischen schwierigen Punkte) konvergierte
das volle, ungesicherte Newton-Verfahren **immer**, allerdings **nie monoton** (der Funktionswert
steigt zwischendurch bei jedem einzigen getesteten Startpunkt). Die **echte, hand-hergeleitete**
Divergenz zeigt sich stattdessen an einer einfacheren Funktion: $f(x)=\ln(1+x^2)$ divergiert
scharf ab genau $|x_0|=1/\sqrt3\approx0{,}5774$.

## Warum dieses Problem

Stück 1 endete mit einer offenen Frage: was, wenn man mehr als nur den Gradienten nutzt? Newton
baut am aktuellen Punkt ein quadratisches Modell aus Gradient UND Hesse-Matrix und springt direkt
zu dessen Minimum. Ist die Zielfunktion selbst eine Quadratik, ist das Modell exakt.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| Newton löst die Quadratik in genau einem Schritt, unabhängig von κ | ✅ 9 von 9 Stichproben (κ=2,20,200), $f_{final}<10^{-20}$ |
| Ersetzt man die Hesse-Matrix durch $(1/\eta)I$, reproduziert Newton exakt Gradientenabstieg | ✅ max. Abweichung $8{,}3\cdot10^{-17}$ (Maschinengenauigkeit) |
| Quadratische Konvergenz nahe dem Optimum (Rosenbrock) | ✅ Exponent 1,85–1,98 bei drei Startpunkten |
| Gradienten-/Hesse-Check gegen finite Differenzen unter $10^{-6}$ | ✅ alle vier Werte zwischen $10^{-10}$ und $10^{-11}$ |
| ⚠️ **Widerlegt:** volles Newton scheitert bei manchen Startpunkten fern vom Optimum auf Rosenbrock | ❌ **Konvergiert bei 34/34 getesteten Startpunkten** (30 Zufallspunkte in [−2, 2]² plus 4 klassische) – aber bei 34/34 NICHT monoton (f steigt zwischendurch) |
| Echte Divergenz zeigt sich stattdessen an $\ln(1+x^2)$ | ✅ scharfe Schwelle exakt bei $|x_0|=1/\sqrt3$ |

## Befunde (gemessen, keine Behauptungen)

**Ein-Schritt-Konvergenz auf der Quadratik** (3 Stichproben je κ, Dimension 5):

| κ | Iterationen | $f$ am Ende |
|---|---|---|
| 2 | 1 | $\approx10^{-31}$ |
| 20 | 1 | $\approx10^{-30}$ |
| 200 | 1 | $\approx10^{-29}$ |

**Quadratische Konvergenz nahe dem Optimum** (Rosenbrock, Fehler $\|x_k-x^*\|$ je Schritt):

| Start | Fehlerfolge | Letzter Exponent |
|---|---|---|
| (0,9; 0,8) | 0,224 → 0,146 → 0,0297 → 0,0101 → $1{,}3\cdot10^{-4}$ → $4{,}6\cdot10^{-7}$ → $2{,}8\cdot10^{-13}$ → 0 | 1,98 |
| (1,1; 1,3) | 0,316 → 0,247 → 0,0096 → 0,0011 → $2{,}4\cdot10^{-7}$ → $5{,}8\cdot10^{-13}$ | 1,85 |
| (0,5; 0,2) | 0,943 → 0,838 → 0,376 → 0,269 → 0,0177 → 0,0016 → $5{,}3\cdot10^{-7}$ → $1{,}7\cdot10^{-12}$ | 1,88 |

Die Zahl der Nachkommastellen verdoppelt sich sichtbar (Exponent nähert sich 2, nicht 1 wie bei
linearer Konvergenz).

**Rosenbrock, 34 Startpunkte (30 Zufallspunkte in [−2, 2]², dazu die klassischen schwierigen Startwerte
$(-1{,}2;\,1{,}0)$, $(-1{,}5;\,-1{,}0)$, $(2{,}0;\,-1{,}0)$, $(-2{,}0;\,2{,}0)$):**

| Konvergiert | Nicht-monoton (f steigt zwischendurch) | Max. Iterationen gebraucht |
|---|---|---|
| 34/34 | 34/34 | 9 |

**Echte Divergenz: $f(x)=\ln(1+x^2)$** (Fixpunkt-Iteration $x_{k+1}=\dfrac{2x_k^3}{x_k^2-1}$,
Schwelle exakt bei $|x_0|=1/\sqrt3\approx0{,}57735$):

| $x_0$ | Abstand zur Schwelle | Divergiert? |
|---|---|---|
| 0,56735 | −0,01 | Nein (6 Schritte) |
| 0,57635 | −0,001 | Nein (8 Schritte) |
| 0,57725 | −0,0001 | Nein (10 Schritte) |
| 0,57745 | +0,0001 | **Ja** (25 Schritte bis über $10^6$) |
| 0,57835 | +0,001 | **Ja** (22 Schritte) |
| 0,58735 | +0,01 | **Ja** (22 Schritte) |

Die Iterierte wächst dabei ungefähr geometrisch (Verdopplung je Schritt), nicht chaotisch.

## Modell und Verfahren

- `nm_functions.py` – Quadratik/Rosenbrock (eigenständige Kopie aus `gradient-descent-demo`) plus
  die neue Log-Beule $\ln(1+x^2)$; je Funktionswert, Gradient und Hesse-Matrix von Hand.
- `nm_optimizer.py` – `newton_method()` (voll, ungesichert) und eine eigenständige
  Gradientenabstieg-Kopie nur für den Reduktions-Check.
- `nm_evaluation.py` – Ein-Schritt-Check, Reduktions-Check, quadratische Konvergenzrate,
  Rosenbrock-Weitwinkel-Sweep, Divergenzschwelle der Log-Beule, Gradienten-/Hesse-Check.
- `nm_visualization.py` – Plotly: 2D-Kontur+Trajektorie, 1D-Landschaft (Log-Beule), Newton-vs-
  Gradientenabstieg, Divergenz-Balken.

## Was die App zeigt

Funktionswahl (Quadratik/Rosenbrock/Log-Beule), je nach Funktion passende Regler
(Konditionszahl, Startpunkt), max. Iterationen und Seed in der Sidebar; Trajektorie/Landschaft
und Konvergenzkurve für die aktuelle Konfiguration; Newton-vs-Gradientenabstieg als zentraler
Befund; ein "📐"-Abschnitt mit der vollständigen Korrektheits-Kette (Ein-Schritt-Beweis,
Reduktions-Check, Konvergenz-Exponenten, Rosenbrock-Weitwinkel-Sweep, Log-Beule-Divergenzschwelle,
Gradienten-/Hesse-Check).

## Was nicht funktioniert hat / Grenzen

**Echte, substantielle Plan-Korrektur:** Die ursprüngliche Hypothese ("Newton scheitert bei
manchen Startpunkten fern vom Optimum auf Rosenbrock") wurde durch die Vormessung **widerlegt** –
selbst bei Zufallsstarts in [−2, 2]² (in einer Vormessung auch bis Radius 20, deutlich weiter als jeder in der Literatur zitierte
"schwierige" Startpunkt für diese Funktion) konvergierte das volle, ungesicherte Newton-Verfahren
in JEDEM der 34 getesteten Fälle, meist innerhalb weniger Iterationen. Der reale, messbare
Schwachpunkt liegt woanders: der Funktionswert sinkt dabei **nicht monoton** – bei allen 34
Startpunkten stieg $f$ mindestens einmal zwischendurch an, ein Verhalten, das Gradientenabstieg
mit Liniensuche per Konstruktion ausschließt. Die tatsächliche, scharfe Divergenz wurde erst an
einer einfacheren, expliziten Funktion gefunden: $\ln(1+x^2)$, deren Newton-Iteration eine
geschlossene Fixpunkt-Formel hat und exakt bei $|x_0|=1/\sqrt3$ kippt – hier als zusätzliche,
robuste Korrektheits-Kette genutzt statt nur als Warnung zitiert.

**Grenzen:** kein Line-Search/Trust-Region-Sicherungsmechanismus (bewusst weggelassen, um die
Schwäche des reinen Verfahrens sichtbar zu halten – Sicherungen sind ein eigenes, hier nicht
behandeltes Thema). Nur unrestringierte Minimierung.

## Tests

37 Tests, `python -m pytest tests/ -v` (Laufzeit lokal ~2 Sekunden):
- `test_functions.py` – Testfunktionen, Gradienten/Hesse-Matrizen gegen finite Differenzen.
- `test_optimizer.py` – Ein-Schritt-Konvergenz, quadratische Konvergenz, Divergenz/Konvergenz an
  der Log-Beule-Schwelle, Array-Längen-Konsistenz auch bei Divergenz.
- `test_evaluation.py` – Sweep-Funktionen mit billigen Parametern.
- `test_claims.py` – jede Zahl oben nachgerechnet, mit Toleranzband (Modul-Fixtures für die
  teureren Sweeps).
- `test_presets.py`, `test_app.py` – Presets, Funktionswechsel, Footer.
- `test_oracle_newton.py` – unabhängige Orakel: Cramersche-Regel-Iteration auf Rosenbrock, `scipy.optimize.rosen*`,
  Fixpunkt-Iteration der Log-Beule (Schwelle und Schrittzahl).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `nm_constants.py` | Regler-Grenzen, Presets |
| `nm_functions.py` | Testfunktionen (Quadratik, Rosenbrock, Log-Beule) |
| `nm_optimizer.py` | Newton-Verfahren, Gradientenabstieg-Kopie |
| `nm_evaluation.py` | Sweeps, Korrektheits-Kette, Gradienten-/Hesse-Check |
| `nm_visualization.py` | Plotly-Plots |
| `nm_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Kein Line-Search/Trust-Region (das würde die Schwäche des reinen Verfahrens verdecken, die genau
der Punkt dieses Stücks ist). Kein Vergleich gegen SciPy – die Korrektheit wird gegen das
**bekannte exakte Optimum** und eine **von Hand hergeleitete Divergenzschwelle** geprüft, eine
stärkere Garantie als ein Solver-Kreuzvergleich.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization* (2. Aufl.). Springer.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Nichtlineare Optimierung: acht Stücke, zwei Äste](https://sebastianhanisch.net/konzepte-nichtlineare-optimierung.html).
