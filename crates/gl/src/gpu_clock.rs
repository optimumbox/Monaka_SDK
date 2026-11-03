//! The GPU's own clock around each frame at `SwapBuffers` (`GL_TIMESTAMP` queries, read frames
//! later so nothing waits): how long the GPU spent from the end of our work at one swap to the
//! start of it at the next (the game's frame on the GPU, idle gaps included), our own work at the
//! swap, and how far the GPU runs behind the CPU (from the CPU reaching a swap to the GPU reaching
//! it: the frames queued ahead of the screen).

use crate::{Result, load};

const GL_TIMESTAMP: u32 = 0x8E28;
const GL_QUERY_RESULT: u32 = 0x8866;
const GL_QUERY_RESULT_AVAILABLE: u32 = 0x8867;
/// Frames whose queries can be in flight at once; an older one not read yet is dropped.
const RING: usize = 8;

type GenQueries = unsafe extern "system" fn(i32, *mut u32);
type DeleteQueries = unsafe extern "system" fn(i32, *const u32);
type QueryCounter = unsafe extern "system" fn(u32, u32);
type GetQueryObjectiv = unsafe extern "system" fn(u32, u32, *mut i32);
type GetQueryObjectui64v = unsafe extern "system" fn(u32, u32, *mut u64);
type GetInteger64v = unsafe extern "system" fn(u32, *mut i64);

/// One frame's times on the GPU (milliseconds).
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuFrame {
    /// From our work's end at the swap before to its start at this one (none after a dropped frame).
    pub game: Option<f64>,
    /// Our own work at this swap.
    pub ours: f64,
    /// From the CPU reaching this swap to the GPU reaching it.
    pub behind: f64,
}

#[derive(Clone, Copy, Default)]
struct Slot {
    start: u32,
    end: u32,
    /// The GPU clock when the CPU marked the start.
    cpu_at: i64,
}

pub struct GpuClock {
    generate: GenQueries,
    delete: DeleteQueries,
    counter: QueryCounter,
    get_iv: GetQueryObjectiv,
    get_u64: GetQueryObjectui64v,
    get_i64: GetInteger64v,
    slots: [Slot; RING],
    next: usize,
    /// The last read frame's end on the GPU, and its slot's number.
    last_end: Option<(u64, u64)>,
    marked: u64,
    read: u64,
    pub dropped: u64,
}

impl GpuClock {
    /// The queries made; the game's GL context must be current.
    pub fn new() -> Result<Self> {
        let mut clock = Self {
            generate: load("glGenQueries")?,
            delete: load("glDeleteQueries")?,
            counter: load("glQueryCounter")?,
            get_iv: load("glGetQueryObjectiv")?,
            get_u64: load("glGetQueryObjectui64v")?,
            get_i64: load("glGetInteger64v")?,
            slots: [Slot::default(); RING],
            next: 0,
            last_end: None,
            marked: 0,
            read: 0,
            dropped: 0,
        };
        let mut names = [0u32; RING * 2];
        // SAFETY: fills `names` with RING * 2 new query names on the current context.
        unsafe { (clock.generate)(names.len() as i32, names.as_mut_ptr()) };
        for (slot, pair) in clock.slots.iter_mut().zip(names.chunks(2)) {
            (slot.start, slot.end) = (pair[0], pair[1]);
        }
        Ok(clock)
    }

    /// Before our work at a swap: the frames whose times are in, read (oldest first), and this
    /// one's start marked.
    pub fn start(&mut self, mut each: impl FnMut(GpuFrame)) {
        while self.read < self.marked {
            let index = (self.read % RING as u64) as usize;
            let slot = self.slots[index];
            let mut available = 0i32;
            // SAFETY: a query this clock made and issued.
            unsafe { (self.get_iv)(slot.end, GL_QUERY_RESULT_AVAILABLE, &mut available) };
            if available == 0 {
                break;
            }
            let (mut start, mut end) = (0u64, 0u64);
            // SAFETY: as above; both results are in once the later one is.
            unsafe {
                (self.get_u64)(slot.start, GL_QUERY_RESULT, &mut start);
                (self.get_u64)(slot.end, GL_QUERY_RESULT, &mut end);
            }
            let ms = |ns: i64| ns as f64 / 1e6;
            let game = self.last_end.filter(|&(_, n)| n + 1 == self.read).map(|(last, _)| ms(start as i64 - last as i64));
            each(GpuFrame { game, ours: ms(end as i64 - start as i64), behind: ms(start as i64 - slot.cpu_at) });
            self.last_end = Some((end, self.read));
            self.read += 1;
        }
        // The ring full: the oldest is given up rather than waited for.
        if self.marked - self.read >= RING as u64 {
            self.read += 1;
            self.dropped += 1;
        }
        let slot = &mut self.slots[self.next];
        // SAFETY: a query this clock made; the clock read on the current context.
        unsafe {
            (self.counter)(slot.start, GL_TIMESTAMP);
            (self.get_i64)(GL_TIMESTAMP, &mut slot.cpu_at);
        }
    }

    /// After our work at the swap: its end marked.
    pub fn end(&mut self) {
        let slot = &mut self.slots[self.next];
        // SAFETY: as in `start`.
        unsafe { (self.counter)(slot.end, GL_TIMESTAMP) };
        self.next = (self.next + 1) % RING;
        self.marked += 1;
    }

    /// The queries freed; the context they were made on must be current.
    pub fn release(self) {
        let names: Vec<u32> = self.slots.iter().flat_map(|s| [s.start, s.end]).collect();
        // SAFETY: names this clock made.
        unsafe { (self.delete)(names.len() as i32, names.as_ptr()) };
    }
}
