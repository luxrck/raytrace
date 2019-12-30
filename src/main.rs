#![feature(box_syntax)]

use std::env;
use std::f64;

use image::RgbImage;
use nalgebra::Vector3;
use rand::prelude::*;
use rayon::prelude::*;

#[derive(Copy, Clone, Debug)]
struct Ray {
    origin: Vector3<f64>,
    direction: Vector3<f64>,
    color: Vector3<f64>,
}

impl Ray {
    fn new(o: Vector3<f64>, d: Vector3<f64>, c: Option<Vector3<f64>>) -> Self {
        let c = match c {
            Some(c) => c,
            None => Vector3::new(1.0, 1.0, 1.0),
        };
        Ray {
            origin: o,
            direction: d.normalize(),
            color: c,
        }
    }

    fn point(&self, t: f64) -> Vector3<f64> {
        self.origin + t * self.direction
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray) -> Option<Ray>;
}

struct World {
    objects: Vec<Box<dyn Hittable>>,
}

impl Hittable for World {
    fn hit(&self, ray: &Ray) -> Option<Ray> {
        let mut nearest: Option<Ray> = None;
        for obj in &self.objects {
            nearest = match obj.hit(ray) {
                Some(oray) => {
                    let nray = match nearest {
                        Some(nray) => {
                            if (oray.origin - nray.origin).dot(&ray.direction) < 0.0 {
                                oray
                            } else {
                                nray
                            }
                        }
                        None => oray,
                    };
                    Some(nray)
                }
                None => continue,
            }
        }

        nearest
    }
}

unsafe impl Send for World {}
unsafe impl Sync for World {}

trait Material {
    fn scatter(&self, normal: Vector3<f64>, in_ray: &Ray) -> Ray;
}

#[derive(Copy, Clone)]
struct Lamberitian {
    albedo: Vector3<f64>,
}

#[derive(Copy, Clone)]
struct Metal {
    albedo: Vector3<f64>,
    fuzz: f64,
}

impl Material for Lamberitian {
    fn scatter(&self, normal: Vector3<f64>, in_ray: &Ray) -> Ray {
        let mut rng = rand::thread_rng();
        let normal = normal.normalize();

        let direction = loop {
            let d = Vector3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>());
            let p = 2.0 * d - Vector3::new(1.0, 1.0, 1.0);
            if p.dot(&p) < 1.0 {
                break normal + p;
            }
        };

        Ray::new(
            Vector3::new(0.0, 0.0, 0.0),
            direction,
            Some(self.albedo.component_mul(&in_ray.color)),
        )
    }
}

impl Material for Metal {
    fn scatter(&self, normal: Vector3<f64>, in_ray: &Ray) -> Ray {
        let mut rng = rand::thread_rng();
        let normal = normal.normalize();
        let in_direction = in_ray.direction;

        // reflection vector direction
        let mut direction = in_direction - 2.0 * in_direction.dot(&normal) * normal;

        let rand_direction = loop {
            let d = Vector3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>());
            let p = 2.0 * d - Vector3::new(1.0, 1.0, 1.0);
            if p.dot(&p) < 1.0 {
                break normal + p;
            }
        };
        direction += self.fuzz * rand_direction;
        Ray::new(
            Vector3::new(0.0, 0.0, 0.0),
            direction,
            Some(self.albedo.component_mul(&in_ray.color)),
        )
    }
}

struct Sphere<M: Material> {
    center: Vector3<f64>,
    radius: f64,
    material: M,
}

impl<M: Material> Hittable for Sphere<M> {
    fn hit(&self, ray: &Ray) -> Option<Ray> {
        let oc = ray.origin - self.center;
        let a: f64 = ray.direction.dot(&ray.direction);
        let b: f64 = oc.dot(&ray.direction);
        let c: f64 = oc.dot(&oc) - self.radius * self.radius;
        let delta = b * b - a * c;

        if delta < 0.0 {
            return None;
        }

        let hitted = -b - delta.sqrt() / a;

        // in_ray hits object A -> calc t -> calc hit point p -> compute reflection ray oray
        // when we compute t for reflection ray on A, is it possible we get an non-zero
        // value t due to loss of significance when compute p ?
        if hitted < 0.00001 {
            return None;
        }

        let point = ray.point(hitted);
        let normal = point - self.center;
        let mut oray = self.material.scatter(normal, ray);

        oray.origin = point;
        Some(oray)
    }
}

unsafe impl<M: Material> Send for Sphere<M> {}
unsafe impl<M: Material> Sync for Sphere<M> {}

struct Camera {
    eye: Vector3<f64>,
    xaxis: Vector3<f64>,
    yaxis: Vector3<f64>,
    origin: Vector3<f64>,
    sampling: u32,
    len_radius: f64,
}

impl Camera {
    fn new(eye: Vector3<f64>, center: Vector3<f64>, xlen: f64, ylen: f64, aperture: f64) -> Self {
        let vcenter = center - eye;
        let zdepth = vcenter.dot(&vcenter).sqrt();
        let vxaxis = xlen * zdepth * vcenter.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize();
        let vyaxis = ylen * zdepth * vxaxis.cross(&vcenter).normalize();
        let vorigin = eye + vcenter - vxaxis / 2.0 - vyaxis / 2.0;

        Camera {
            eye: eye,
            xaxis: vxaxis,
            yaxis: vyaxis,
            origin: vorigin,
            sampling: 0,
            len_radius: aperture / 2.0,
        }
    }

    fn with_sampling(mut self, samples: u32) -> Self {
        self.sampling = samples;
        self
    }

    fn ray(&self, u: f64, v: f64) -> Ray {
        let mut rng = rand::thread_rng();
        let rand_direction = self.len_radius
            * loop {
                let d = Vector3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>());
                let p = 2.0 * d - Vector3::new(1.0, 1.0, 1.0);
                if p.dot(&p) < 1.0 {
                    break p;
                }
            };
        let offset = Vector3::new(rand_direction.x, 0.0, 0.0)
            .component_mul(&self.xaxis.normalize())
            + Vector3::new(0.0, rand_direction.y, 0.0).component_mul(&self.yaxis.normalize());
        Ray::new(
            self.eye + offset,
            self.origin + u * self.xaxis + v * self.yaxis - self.eye - offset,
            None,
        )
    }

    fn trace(&self, ray: Ray, world: &World) -> Vector3<f64> {
        let sky_color = |d: &Vector3<f64>| {
            let sky = Vector3::new(0.5, 0.7, 1.0);
            let t = 0.5 * (d.y + 1.0);
            (1.0 - t) * Vector3::new(1.0, 1.0, 1.0) + t * sky
        };

        let mut iray = ray;
        let max_trace = 50;
        let mut current_trace = 0;

        let color = loop {
            iray = match world.hit(&iray) {
                Some(nray) => nray,
                None => {
                    break iray.color.component_mul(&sky_color(&iray.direction));
                }
            };

            current_trace += 1;
            if current_trace > max_trace {
                break Vector3::new(0.0, 0.0, 0.0);
            }
        };

        color
    }

    fn render(&self, world: &World, resolution: (u32, u32)) -> Option<RgbImage> {
        let (width, height) = resolution;

        let render_pixel = |x: u32, y: u32, rng: &mut ThreadRng| {
            let color = if self.sampling > 0 {
                let mut color = Vector3::new(0.0, 0.0, 0.0);
                for _ in 0..self.sampling {
                    let u: f64 = (x as f64 + rng.gen::<f64>()) / (width as f64);
                    let v: f64 = (y as f64 + rng.gen::<f64>()) / (height as f64);
                    color += self.trace(self.ray(u, v), world);
                }

                color /= self.sampling as f64;
                color
            } else {
                let u: f64 = (x as f64) / (width as f64);
                let v: f64 = (y as f64) / (height as f64);
                self.trace(self.ray(u, v), world)
            };

            let r: u8 = (255.99 * color.x.sqrt()) as u8;
            let g: u8 = (255.99 * color.y.sqrt()) as u8;
            let b: u8 = (255.99 * color.z.sqrt()) as u8;

            vec![r, g, b]
        };

        let buf: Vec<_> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut rng = rand::thread_rng();
                (0..width)
                    .into_iter()
                    .map(|x| render_pixel(x, height - y, &mut rng))
                    .flatten()
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();

        RgbImage::from_raw(width, height, buf)
    }
}

fn main() {
    let eye = Vector3::new(-1.0, 3.0, 3.0);
    let center = Vector3::new(0.0, 0.0, -1.0);
    let cam = Camera::new(eye, center, 0.8, 0.4, 2.0).with_sampling(256);

    let diffuse = Lamberitian {
        albedo: Vector3::new(0.8, 0.3, 0.3),
    };
    let diffuse1 = Lamberitian {
        albedo: Vector3::new(0.8, 0.8, 0.0),
    };
    let metal = Metal {
        albedo: Vector3::new(0.8, 0.8, 0.8),
        fuzz: 0.4,
    };
    let metal1 = Metal {
        albedo: Vector3::new(0.8, 0.6, 0.2),
        fuzz: 1.0,
    };
    let metal2 = Metal {
        albedo: Vector3::new(0.6, 0.6, 0.6),
        fuzz: 0.1,
    };

    let s1 = Sphere {
        center: Vector3::new(-1.0, 0.0, -1.0),
        radius: 0.5,
        material: metal,
    };
    let s2 = Sphere {
        center: Vector3::new(0.0, 0.0, -1.0),
        radius: 0.5,
        material: diffuse,
    };
    let s3 = Sphere {
        center: Vector3::new(1.0, 0.0, -1.0),
        radius: 0.5,
        material: metal1,
    };
    let s4 = Sphere {
        center: Vector3::new(0.0, -100.5, -1.0),
        radius: 100.0,
        material: diffuse1,
    };
    let s5 = Sphere {
        center: Vector3::new(0.0, 0.5, -2.5),
        radius: 1.0,
        material: metal2,
    };

    let world = World {
        objects: vec![box s1, box s2, box s3, box s4, box s5],
    };

    let result = cam.render(&world, (1000, 500)).unwrap();
    result.save(env::args().nth(1).unwrap()).unwrap();
}
