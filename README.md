<<<<<<< HEAD
# Slope Stability Analysis

## Description

This package determines the stability of a slope by calculating the factor of safety for the critical slip surface using the stability charts developed by Janbu (1954). The package provides four functions: `CHI_PHI_SOIL`, `INFINITE_SLOPE`, `PURELY_COHESIVE_SOIL`, and `PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH`. Inputs for these functions must be in either SI units (m, kN/m<sup>2</sup> , kN/m<sup>3</sup> ) or FPS units (feet, pcf, psf).

## Installation

This package can be installed using:

```sh
pip install SlopeStability
```

## Usage
The following functions can be used:
### (1) For Purely cohesive soil(Φ=0)
```sh
 PURELY_COHESIVE_SOIL(beta, H, Hw, Hw', D, c, lw, l, Ht, q)
```
beta: Slope Angle
H: Height of slope above toe (feet/m)<br />
Hw: Height of external water level above toe (feet/m)<br />
Hw': Height of internal water level above toe (feet/m)<br />
D: Depth from the toe of the slope to the lowest point on the slip circle<br />
c: Average shear strength (kN/m<sup>2</sup> or psf)<br />
lw: Unit weight of water (kN/m<sup>3</sup> or pcf)<br />
l: Unit weight of soil (kN/m<sup>3</sup> or pcf)<br />
Ht: Depth of tension crack (feet/m)<br />
q: Surcharge (kN/m<sup>2</sup> or psf)<br />

#### Return a list [Factor of safety, x-coordinate, y-corrdinate,Radius of slip circle]
#### Applicable for soil with a friction angle (phi=0).

### (2) For Soil Having non-zero frinciton angle and cohesion(c-phi soil)

```sh
CHI_PHI_SOIL(beta, H, Hw, Hc, Hw', c, &#966, l, lw, q, Ht)
```
beta: Slope Angle<br />
H: Height of slope above toe (feet/m)<br />
Hw: Height of external water level above toe (feet/m)<br />
Hw': Height of internal water level above toe (feet/m)<br />
Hc: Height of internal water level measured beneath the crest of the slope to the lowest point on the slip circle<br />
c: Average shear strength (kN/m<sup>2</sup> or psf)<br />
phi: Frictional angle of slope (degree)<br />
lw: Unit weight of water (kN/m<sup>3</sup> or pcf)<br />
l: Unit weight of soil (kN/m<sup>3</sup> or pcf)<br />
Ht: Depth of tension crack (feet/m)<br />
q: Surcharge (kN/m<sup>2</sup> or psf)<br />

#### Return a list [Factor of safety, x-coordinate, y-corrdinate,Radius of slip circle]
#### Assumption: For phi>0, the critical circle passes through the toe of the slope.

### (3) For Soil having cohesionless material(C=0)
```sh
INFINITE_SLOPE(beta, theeta, H, c, phi, c', phi', l, lw, X, T)
```
beta: Slope Angle (degree)<br />
H: Height of slope above toe (feet/m)<br />
theeta: Angle of seepage measured from the horizontal direction (degree)<br />
c: Cohesion for total stress analysis (kN/m<sup>2</sup> or psf)<br />
phi: Frictional angle for total stress analysis (degree)<br />
c': Cohesion for effective stress analysis (kN/m<sup>2</sup> or psf)<br />
phi': Frictional angle for effective stress analysis (degree)<br />
lw: Unit weight of water (kN/m<sup>3</sup> or pcf)<br />
l: Unit weight of soil (kN/m<sup>3</sup> or pcf)<br />
X: Distance from the depth of sliding to the surface of seepage, measured normal to the surface of the slope (feet/m)<br />
T: Distance from the depth of sliding to the surface of the slope, measured normal to the surface of the slope (feet/m)<br />

#### Return a list [Factor of safety]
#### Applicable for cohesionless materials (c=0) where the critical failure mechanism is shallow sliding or slopes in residual soils, where a relatively thin layer of soil overlies firmer soil or rock, and the critical failure mechanism is sliding along a plane parallel to the slope, at the top of the firm layer.

### (4) For purely cohesive Soil(Φ=0) with linearly increasing shear strength 

```sh
PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH(beta, H, H0, Cb, l, lb)
```
beta: Slope angle<br />
H: Height of the slope above toe (feet/m)<br />
H0: Height at which the straight line of shear strength intersects zero(feet/m)<br />
Cb: Strength at the elevation of the toe of the slope(kN/m<sup>2</sup> or psf)<br />
l: Weighted average unit weight for partly submerged slopes(kN/m<sup>3</sup> or pcf)<br />
lb: Buoyant unit weight for submerged slopes(kN/m<sup>3</sup>or pcf)<br />

#### Return a list [Factor of safety]

## Contributing
#### Contributions are welcome! Please feel free to submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
Name: [Piyush Jangid]<br />
Email: [xyz@gmail.com]<br />
GitHub: yourusername
=======
# Slope-Stability
>>>>>>> origin/main
