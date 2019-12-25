#![feature(box_syntax)]

use std::f64;
use std::env;

use nalgebra::Vector3;
use image::{ RgbImage, ImageBuffer, Rgb };
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
    // fn color(&self, ray: &Ray) -> Vector3<f64>;
}

struct World {
    objects: Vec<Box<dyn Hittable>>,
}

impl Hittable for World {
    fn hit(&self, ray: &Ray) -> Option<Ray> {
        for obj in &self.objects {
            match obj.hit(ray) {
                Some(rr) => return Some(rr),
                None => continue,
            }
        }
        None
    }
}

unsafe impl Send for World {}
unsafe impl Sync for World {}

trait Material {
    fn scatter(&self, normal: Vector3<f64>, in_direction: Vector3<f64>) -> Ray;
}

#[derive(Copy, Clone)]
struct Lamberitian {
    albedo: Vector3<f64>,
}

#[derive(Copy, Clone)]
struct Metal {
    albedo: Vector3<f64>,
}

impl Material for Lamberitian {
    fn scatter(&self, normal: Vector3<f64>, _: Vector3<f64>) -> Ray {
        let mut rng = rand::thread_rng();
        let normal = normal.normalize();

        let direction = loop {
            let d = Vector3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>());
            let p = 2.0 * d - Vector3::new(1.0, 1.0, 1.0);
            if p.dot(&p) < 1.0 {
                break normal + p;
            }
        };

        Ray::new(Vector3::new(0.0, 0.0, 0.0), direction, Some(self.albedo))
    }
}

impl Material for Metal {
    fn scatter(&self, normal: Vector3<f64>, in_direction: Vector3<f64>) -> Ray {
        let normal = normal.normalize();
        let in_direction = in_direction.normalize();

        let direction = 2.0 * normal + in_direction;
        Ray::new(Vector3::new(0.0, 0.0, 0.0), direction, Some(self.albedo))
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
        let b: f64 = 2.0 * oc.dot(&ray.direction);
        let c: f64 = oc.dot(&oc) - self.radius * self.radius;
        let delta = b * b - 4.0 * a * c;

        let (mut tm, mut tu) = (0.0, 0.0);

        if delta < 0.0 {
            return None;
        } else {
            let base = -b / (2.0 * a);
            let tmp = delta.sqrt() / (2.0 * a);
            if delta > 0.0 {
                tm = base - tmp;
                tu = base + tmp;
                if tm > tu {
                    let _t = tm;
                    tm = tu;
                    tu = _t;
                }
            } else {
                tm = base;
                tu = base;
            }
        }

        if tu < 0.0 {
            None
        } else {
            // Difuse Reflection
            let mut rng = rand::thread_rng();
            let point = ray.point(tm);
            let normal = point - self.center;
            let mut oray = self.material.scatter(normal, ray.direction);
            // let direction = loop {
            //     let d = Vector3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>());
            //     let p = 2.0 * d - Vector3::new(1.0, 1.0, 1.0);
            //     if p.dot(&p) < 1.0 {
            //         break point + (point - self.center).normalize() + p
            //     }
            // };
            // let self_color = Vector3::new(0.5, 0.5, 0.55).component_mul(&ray.color);
            // self_color = self_color.dot(ray.color);

            // Some(Ray::new(point, direction, Some(self_color)))
            oray.origin = point;
            Some(oray)
        }
    }
    // fn color(&self, c: Option<Vector3<f64>>) -> Vector3<f64> {
    //     let c = match c {
    //         Some(col) => col,
    //         None => Vector3::new(1.0, 1.0, 1.0)
    //     };

    //     let self_color = Vector3::new(0.7, 0.8, 0.75);

    //     c.dot(&self_color)
    //     // match self.hit(ray) {
    //     //     Some(ray) => {
    //     //         0.5 * self.color(&ray)
    //     //         //let p = ray.point(tm);
    //     //         //let nv= (p - self.center).normalize();
    //     //         //Some(0.5 * (nv + Vector3::new(1.0, 1.0, 1.0)))
    //     //     },
    //     //     None => ray.color(None)
    //     // }
    // }
}

unsafe impl<M: Material> Send for Sphere<M> {}
unsafe impl<M: Material> Sync for Sphere<M> {}

struct Camera {
    eye: Vector3<f64>,
    xaxis: Vector3<f64>,
    yaxis: Vector3<f64>,
    origin: Vector3<f64>,
    antialiasing: u8,
}

impl Camera {
    fn new(eye: Vector3<f64>, xaxis: f64, yaxis: f64, zdepth: f64) -> Self {
        Camera {
            eye: eye,
            xaxis: Vector3::new(xaxis, 0.0, 0.0),
            yaxis: Vector3::new(0.0, yaxis, 0.0),
            origin: Vector3::new(-xaxis / 2.0, -yaxis / 2.0, zdepth),
            antialiasing: 0,
        }
    }

    fn with_antialiasing(mut self, samples: u8) -> Self {
        self.antialiasing = samples;
        self
    }

    fn ray(&self, u: f64, v: f64) -> Ray {
        Ray::new(
            self.eye,
            self.origin + u * self.xaxis + v * self.yaxis,
            None,
        )
    }

    fn trace(&self, ray: Ray, world: &World) -> Vector3<f64> {
        // fn color(ray: &Ray, world: &World) -> Option<Ray> {
        //     for obj in &world.objects {
        //         if let Some(r) = obj.hit(ray) {
        //             return 0.5 * obj.color(ray);
        //         }
        //     }
        // }
        let sky = Vector3::new(0.5, 0.7, 1.0);
        let norm_d = ray.direction.normalize();
        let t = 0.5 * (norm_d.y + 1.0);
        let sky_color = (1.0 - t) * Vector3::new(1.0, 1.0, 1.0) + t * sky;

        let max_trace = 50;
        let mut current_trace = 0;
        // while let Some(ray) = world.hit(ray) {
        //     current_trace += 1;
        //     if current_trace >= max_trace {
        //         break ray.color;
        //     }
        // }

        match world.hit(&ray) {
            Some(ray) => {
                let mut in_ray = ray;
                let mut color = Vector3::new(1.0, 1.0, 1.0);
                loop {
                    // let rr = world.hit(ray);
                    in_ray = match world.hit(&in_ray) {
                        None => break,
                        Some(ray) => {
                            color.component_mul_assign(&in_ray.color);
                            ray
                        }
                    };
                    current_trace += 1;
                    if current_trace >= max_trace {
                        break;
                    }
                }
                color
            }
            None => sky_color,
        }
        // ray.color(None)
        // let (mut t_min, mut t_max) = (f64::INFINITY, f64::INFINITY);
        // let mut obj = &world.objects[0];
        // for ob in &world.objects {
        //     if let Some((tm,tu)) = ob.hit(&ray) {
        //         if tu < 0.0 {
        //             continue;
        //         }
        //         if tu < t_min {
        //             t_min = tm;
        //             t_max = tu;
        //             obj = ob;
        //         } else {
        //             if tm < t_max {
        //                 if tm < t_min {
        //                     t_min = tm;
        //                     obj = ob;
        //                 }
        //                 if tu > t_max {
        //                     t_max = tu;
        //                 }
        //             }
        //         }
        //     }
        // }
        // if t_min.is_infinite() || t_max.is_infinite() {
        //     ray.color(None)
        // } else {
        //     ray.color(Some(obj))
        // }
    }

    fn render(&self, world: &World, resolution: (u32, u32)) -> Option<RgbImage> {
        let (width, height) = resolution;

        let render_pixel = |x: u32, y: u32, rng: &mut ThreadRng| {
            let color = if self.antialiasing > 0 {
                let mut color = Vector3::new(0.0, 0.0, 0.0);
                for _ in 0..self.antialiasing {
                    let u: f64 = (x as f64 + rng.gen::<f64>()) / (width as f64);
                    let v: f64 = (y as f64 + rng.gen::<f64>()) / (height as f64);
                    color += self.trace(self.ray(u, v), world);
                }

                color /= self.antialiasing as f64;
                color
            } else {
                let u: f64 = (x as f64) / (width as f64);
                let v: f64 = (y as f64) / (height as f64);
                self.trace(self.ray(u, v), world)
            };

            let r: u8 = (255.99 * color.x.sqrt()) as u8;
            let g: u8 = (255.99 * color.y.sqrt()) as u8;
            let b: u8 = (255.99 * color.z.sqrt()) as u8;

            vec! [r, g, b]
        };

        let buf: Vec<_> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut rng = rand::thread_rng();
                (0..width)
                    .into_iter()
                    .map(|x| {
                        render_pixel(x, height - y, &mut rng)
                    })
                    .flatten()
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect();
        
        RgbImage::from_raw(width, height, buf)

        // println!("P3");
        // println!("{} {}", columns, rows);
        // println!("255");
        // for j in (0..rows).rev() {
        //     for i in 0..columns {
        //         let color = if self.antialiasing > 0 {
        //             let mut color = Vector3::new(0.0, 0.0, 0.0);
        //             for _ in 0..self.antialiasing {
        //                 let u: f64 = (i as f64 + rng.gen::<f64>()) / (columns as f64);
        //                 let v: f64 = (j as f64 + rng.gen::<f64>()) / (rows as f64);
        //                 color += self.trace(self.ray(u, v), world);
        //             }

        //             color /= self.antialiasing as f64;
        //             color
        //         } else {
        //             let u: f64 = (i as f64) / (columns as f64);
        //             let v: f64 = (j as f64) / (rows as f64);
        //             self.trace(self.ray(u, v), world)
        //         };

        //         let r: u8 = (255.99 * color.x.sqrt()) as u8;
        //         let g: u8 = (255.99 * color.y.sqrt()) as u8;
        //         let b: u8 = (255.99 * color.z.sqrt()) as u8;

        //         println!("{} {} {}", r, g, b);
        //     }
        // }
    }
}

fn main() {
    let eye = Vector3::new(0.0, 0.0, 0.0);
    let cam = Camera::new(eye, 4.0, 2.0, -1.0).with_antialiasing(100);

    let diffuse = Lamberitian {
        albedo: Vector3::new(0.2, 0.2, 0.9),
    };
    let diffuse1 = Lamberitian {
        albedo: Vector3::new(0.8, 0.3, 0.3),
    };
    let metal = Metal {
        albedo: Vector3::new(0.8, 0.8, 0.0),
    };

    let s1 = Sphere {
        center: Vector3::new(-0.5, 0.0, -1.0),
        radius: 0.5,
        material: diffuse,
    };
    let s2 = Sphere {
        center: Vector3::new(0.0, -100.5, -1.0),
        radius: 100.0,
        material: diffuse,
    };
    let s3 = Sphere {
        center: Vector3::new(0.5, 0.0, -1.0),
        radius: 0.5,
        material: metal,
    };

    // let ray = Ray::new(eye, Vector3::new(0.0,1.0,-1.0), None);
    // let ra = s2.hit(&ray);
    // println!("{:?}", ra);
    let world = World {
        objects: vec![box s1, box s2, box s3],
    };

    // let ray = Ray::new(eye, Vector3::new(0.0, 0.0, -1.0));
    // let c = cam.nearest(&ray, &world);
    // println!("{:?}", c);
    let result = cam.render(&world, (200, 100)).unwrap();
    result.save(env::args().nth(1).unwrap()).unwrap();
}
