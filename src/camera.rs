use crate::math::{Point3d, Vec3f};
use winit_input_helper::WinitInputHelper;
use winit::event::VirtualKeyCode;

#[derive(Debug)]
pub struct Camera {
    pub loc: Point3d,
    pub vel: Vec3f,
    pub acc: Vec3f,
    pub yaw: f64,
    pub pitch: f64,
    speed: f64,
}

impl Camera {
    const WORLD_UP: Vec3f = Vec3f {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    const WALK_SPEED: f64 = 5.0;
    const SENS: f64 = 0.075;

    pub fn new(
        loc: Point3d,
        yaw: f64,
        pitch: f64,
    ) -> Self {
        Camera {
            loc,
            vel: Vec3f::default(),
            acc: Vec3f::default(),
            yaw,
            pitch,
            speed: Self::WALK_SPEED,
        }
    }

    pub fn update_pos(&mut self, dt: f64, input: &WinitInputHelper) {
        self.update_acc(input);
        self.update_vel();
        self.update_loc(dt);
    }

    fn update_loc(&mut self, dt: f64) {
        let s = self.speed;
        let v = &self.vel;

        self.loc.x += s * v.x * dt;
        self.loc.y += s * v.y * dt;
        self.loc.z += s * v.z * dt;
    }

    fn update_vel(&mut self) {
        let forward = Vec3f {
            x: self.yaw.to_radians().sin() * self.pitch.to_radians().cos(),
            y: self.pitch.to_radians().sin(),
            z: -(self.yaw.to_radians().cos() * self.pitch.to_radians().cos()),
        }.normalize();
        let right = Vec3f::cross(&forward, &Camera::WORLD_UP).normalize();

        let forward = Vec3f::new(forward.x, 0.0, forward.z).normalize();
        let right = Vec3f::new(right.x, 0.0, right.z).normalize();

        self.vel.x = (self.acc.z * forward.x) + (self.acc.x * right.x);
        self.vel.y = self.acc.y;
        self.vel.z = (self.acc.z * forward.z) + (self.acc.x * right.z);
    }

    fn update_acc(&mut self, input: &WinitInputHelper) {
        self.acc = Vec3f::default();
        if input.key_held(VirtualKeyCode::W) {
            self.acc.z += 1.0;
        }
        if input.key_held(VirtualKeyCode::S) {
            self.acc.z -= 1.0;
        }
        if input.key_held(VirtualKeyCode::D) {
            self.acc.x += 1.0;
        }
        if input.key_held(VirtualKeyCode::A) {
            self.acc.x -= 1.0;
        }
        if input.key_held(VirtualKeyCode::Space) {
            self.acc.y += 1.0;
        }
        if input.key_held(VirtualKeyCode::LShift) {
            self.acc.y -= 1.0;
        }
    }

    pub fn update_look(&mut self, look: (f64, f64)) {
        self.yaw += Self::SENS * look.0;
        self.pitch += Self::SENS * -look.1;

        if self.yaw > 360.0 {
            self.yaw = 0.0;
        }
        if self.yaw < 0.0 {
            self.yaw = 360.0;
        }
        if self.pitch > 89.99 {
            self.pitch = 89.99;
        }
        if self.pitch < -89.99 {
            self.pitch = -89.99;
        }
    }
}