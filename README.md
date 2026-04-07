# Simulare Entropie - Gaz Ideal în 3D

Acest proiect demonstrează principiile termodinamicii și creșterea entropiei (A Doua Lege a Termodinamicii) folosind o simulare interactivă 3D. Particulele, reprezentând moleculele unui gaz, pornesc dintr-o stare ordonată (entropie minimă, concentrate într-un colț al unui recipient) și se extind treptat ocupând întreg volumul, atingând o stare de echilibru termodinamic (entropie maximă).

## Concepte Fizice Ilustrate
1. **Entropia ca Măsură a Dezordinii:** Sistemul evoluează natural de la o stare cu probabilitate mică (toate particulele într-un colț) la starea cu probabilitate maximă (distribuție uniformă).
2. **Formula Entropiei:** Entropia totală este calculată pe baza numărului de stări microscopice ($W$) prin relația lui Boltzmann: $S = k_B \ln(W)$.
3. **Fluctuații Termice:** Chiar și la echilibru, entropia fluctuează ușor pe măsură ce particulele se mișcă aleatoriu între compartimentele invizibile (celulele de calcul).

## Tehnologii și Librării Folosite
- **[Taichi (ti)](https://www.taichi-lang.org/):** Folosit pentru accelerarea pe GPU a calculelor fizice (poziții, viteze, numărare). Permite simularea eficientă a mii de particule în timp real.
- **[NumPy (np)](https://numpy.org/):** Utilizat pentru manipularea vectorilor și extragerea structurilor de date matematice.
- **[PyVista (pv) & PyVistaQt](https://docs.pyvista.org/):** Folosite pentru randarea 3D complexă, performantă și interactivă. PyVista gestionează shader-ele și afișarea gridului care își schimbă culoarea (coolwarm) în funcție de concentrația locală.
- **[SciPy (scipy.special.gammaln)](https://scipy.org/):** Utilizat pentru a calcula rapid factorialele matematice mari folosind logaritmul funcției Gamma ($\ln(N!)$), o etapă esențială în calcularea entropiei statistice.

## Rularea Proiectului

```bash
# Asigurați-vă că aveți instalate librăriile necesare:
pip install numpy taichi pyvista pyvistaqt scipy matplotlib PyQt5

# Rularea simulării:
python simulare_entropie01.py
```

## Cum se folosește
* Odată pornit scriptul, se va deschide o fereastră 3D (BackgroundPlotter).
* Poți roti spațiul 3D dând click și trăgând cu mouse-ul (stânga).
* Poți da zoom cu rotița (scroll) sau click-dreapta + drag.
* Pentru a opri simularea, apasă butonul roșu din colțul stânga-jos al ecranului simulării sau închide fereastra în mod clasic.
