# PytesiMINT

**Pytesi**mal **M**etal **INT**rusion model

A Python-powered 3D thermal evolution model of the pallasite formation region in the palalsite parent body. Designed to be used with parent-body background mantle temperatures from [Pytesimal](https://pytesimal.readthedocs.io/en/latest/) [1].

PytesiMINT uses the semi-implicit fraction-step method to apply Crank-Nicolson to 3D [2, 3]. Spatial diffusivity is incorporated and the heat capacity method tracks latent heat during crystallisation of the metal.

While PytesiMINT is designed to model the formation of pallasite meteorites, specifically the cooling following injection of molten metal into the solid olivine mantle of the parent body, it can be applied to any conductively-cooling intrusion problem, especially those where spatially varying diffusivity between the intrusive material and country rock is relevant.

The following web resources were useful in the initial development stages of this project:
- [The Crank-Nicolson method implemented from scratch in Python](https://georg.io/2013/12/03/Crank_Nicolson), Georg Walther (2013). [Archived version, 2023-03-27.](https://web.archive.org/web/20230327142215/https://georg.io/2013/12/03/Crank_Nicolson)
- [Python implementation of Crank-Nicolson scheme](http://www.claudiobellei.com/2016/11/10/crank-nicolson/), Claudio Bellei (2019). [Archived version, 2023-03-27.](https://web.archive.org/web/20200725120318/http://www.claudiobellei.com/2016/11/10/crank-nicolson/)
- [Practical Numerical Methods with Python](https://notebook.community/cmitR/numerical-mooc/lessons/04_spreadout/04_05_Crank-Nicolson),  L.A. Barba, C.D. Cooper, G.F. Forsyth (2014). [Archived version, 2023-03-27.](https://web.archive.org/web/20230327152231/https://notebook.community/cmitR/numerical-mooc/lessons/04_spreadout/04_05_Crank-Nicolson)


1. Murphy Quinlan, M., Walker, A. M., Davies, C. J., Mound, J. E., Müller, T., & Harvey, J. (2021). The conductive cooling of planetesimals with temperature-dependent properties. Journal of Geophysical Research: Planets, 126, e2020JE006726. https://doi.org/10.1029/2020JE006726

2. W. Cen, R. Hoppe, N. Gu. Fast and accurate determination of 3D temperature distribution using fraction-step semi-implicit method
AIP Adv., 6 (2016), Article 095305, https://doi.org/10.1063/1.4962665

3. Sandeep Sahijpal, Thermal evolution of non-spherical asteroids in the early solar system, Icarus, Volume 362, (2021). https://doi.org/10.1016/j.icarus.2021.114439.
