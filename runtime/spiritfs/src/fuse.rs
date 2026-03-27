//! FUSE filesystem implementation for SpiritFS.
//!
//! Implements the `fuser::Filesystem` trait, providing POSIX-compatible file
//! operations that passthrough to the backing filesystem while asynchronously
//! queuing semantic indexing jobs.
//!
//! // PHASE1-SHIM: Uses FUSE via the `fuser` crate. Phase 2+ integrates
//! // directly with the SpiritKernel VFS layer.

#![cfg(feature = "fuse")]

use std::collections::HashMap;
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossbeam_channel::Sender;
use fuser::{
    FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory,
    ReplyEmpty, ReplyEntry, ReplyOpen, ReplyWrite, Request,
};
use parking_lot::RwLock;
use tracing::{debug, error, warn};

use crate::metrics::SpiritFsMetrics;
use crate::types::IndexJob;

/// TTL for cached attributes (1 second).
const TTL: Duration = Duration::from_secs(1);

/// Root inode number.
const ROOT_INO: u64 = 1;

/// FUSE filesystem for SpiritFS.
///
/// Implements POSIX operations via passthrough to the backing directory,
/// while queuing write events for async semantic indexing.
pub struct SpiritFuse {
    /// Path to the backing filesystem directory.
    backing_dir: PathBuf,
    /// Channel for submitting index jobs (never blocks POSIX ops).
    index_tx: Sender<IndexJob>,
    /// Mount point path.
    mount_point: PathBuf,
    /// Metrics collector.
    metrics: Arc<SpiritFsMetrics>,
    /// Inode-to-path mapping (cached).
    inode_map: RwLock<HashMap<u64, PathBuf>>,
    /// Next inode number to allocate.
    next_ino: RwLock<u64>,
    /// Open file handles: fh → backing file descriptor info.
    open_files: RwLock<HashMap<u64, OpenFile>>,
    /// Next file handle number.
    next_fh: RwLock<u64>,
}

/// An open file handle.
struct OpenFile {
    path: PathBuf,
    #[allow(dead_code)]
    flags: i32,
}

impl SpiritFuse {
    /// Create a new FUSE filesystem.
    pub fn new(
        backing_dir: PathBuf,
        mount_point: PathBuf,
        index_tx: Sender<IndexJob>,
        metrics: Arc<SpiritFsMetrics>,
    ) -> Self {
        let mut inode_map = HashMap::new();
        inode_map.insert(ROOT_INO, PathBuf::new());

        Self {
            backing_dir,
            index_tx,
            mount_point,
            metrics,
            inode_map: RwLock::new(inode_map),
            next_ino: RwLock::new(2),
            open_files: RwLock::new(HashMap::new()),
            next_fh: RwLock::new(1),
        }
    }

    /// Mount the filesystem and block until unmounted.
    pub fn mount(self, mount_point: &Path, allow_other: bool) -> std::io::Result<()> {
        let mut options = vec![
            MountOption::FSName("spiritfs".to_string()),
            MountOption::AutoUnmount,
            MountOption::DefaultPermissions,
        ];
        if allow_other {
            options.push(MountOption::AllowOther);
        }

        fuser::mount2(self, mount_point, &options)
    }

    /// Resolve an inode to its backing filesystem path.
    fn inode_to_backing_path(&self, ino: u64) -> Option<PathBuf> {
        let map = self.inode_map.read();
        map.get(&ino).map(|rel| self.backing_dir.join(rel))
    }

    /// Resolve an inode to its mount-relative path.
    fn inode_to_mount_path(&self, ino: u64) -> Option<PathBuf> {
        let map = self.inode_map.read();
        map.get(&ino).map(|rel| self.mount_point.join(rel))
    }

    /// Allocate a new inode number.
    fn alloc_ino(&self) -> u64 {
        let mut next = self.next_ino.write();
        let ino = *next;
        *next += 1;
        ino
    }

    /// Allocate a new file handle.
    fn alloc_fh(&self) -> u64 {
        let mut next = self.next_fh.write();
        let fh = *next;
        *next += 1;
        fh
    }

    /// Register an inode for a relative path.
    fn register_inode(&self, rel_path: PathBuf) -> u64 {
        // Check if already registered
        {
            let map = self.inode_map.read();
            for (&ino, path) in map.iter() {
                if *path == rel_path {
                    return ino;
                }
            }
        }

        let ino = self.alloc_ino();
        self.inode_map.write().insert(ino, rel_path);
        ino
    }

    /// Convert std::fs::Metadata to fuser::FileAttr.
    fn meta_to_attr(&self, ino: u64, meta: &std::fs::Metadata) -> FileAttr {
        let kind = if meta.is_dir() {
            FileType::Directory
        } else if meta.is_symlink() {
            FileType::Symlink
        } else {
            FileType::RegularFile
        };

        let atime = meta.accessed().unwrap_or(UNIX_EPOCH);
        let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
        let ctime = mtime; // No creation time on Linux, use mtime

        use std::os::unix::fs::MetadataExt;
        FileAttr {
            ino,
            size: meta.len(),
            blocks: meta.blocks(),
            atime,
            mtime,
            ctime,
            crtime: UNIX_EPOCH,
            kind,
            perm: meta.mode() as u16,
            nlink: meta.nlink() as u32,
            uid: meta.uid(),
            gid: meta.gid(),
            rdev: meta.rdev() as u32,
            blksize: meta.blksize() as u32,
            flags: 0,
        }
    }

    /// Queue an index job without blocking.
    fn queue_index(&self, path: PathBuf) {
        let _ = self.index_tx.try_send(IndexJob::Update(path));
    }
}

impl Filesystem for SpiritFuse {
    fn lookup(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEntry) {
        self.metrics.fuse_lookups_total.inc();

        let parent_backing = match self.inode_to_backing_path(parent) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let child_backing = parent_backing.join(name);
        match std::fs::metadata(&child_backing) {
            Ok(meta) => {
                let parent_rel = self
                    .inode_map
                    .read()
                    .get(&parent)
                    .cloned()
                    .unwrap_or_default();
                let child_rel = parent_rel.join(name);
                let ino = self.register_inode(child_rel);
                let attr = self.meta_to_attr(ino, &meta);
                reply.entry(&TTL, &attr, 0);
            }
            Err(_) => {
                reply.error(libc::ENOENT);
            }
        }
    }

    fn getattr(&mut self, _req: &Request, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        let backing = match self.inode_to_backing_path(ino) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        match std::fs::metadata(&backing) {
            Ok(meta) => {
                let attr = self.meta_to_attr(ino, &meta);
                reply.attr(&TTL, &attr);
            }
            Err(_) => {
                reply.error(libc::ENOENT);
            }
        }
    }

    fn readdir(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        let backing = match self.inode_to_backing_path(ino) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let entries = match std::fs::read_dir(&backing) {
            Ok(e) => e,
            Err(_) => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let mut full_entries = vec![
            (ino, FileType::Directory, ".".to_string()),
            (ino, FileType::Directory, "..".to_string()),
        ];

        let parent_rel = self.inode_map.read().get(&ino).cloned().unwrap_or_default();

        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let child_rel = parent_rel.join(&name);
            let child_ino = self.register_inode(child_rel);

            let kind = if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                FileType::Directory
            } else if entry.file_type().map(|t| t.is_symlink()).unwrap_or(false) {
                FileType::Symlink
            } else {
                FileType::RegularFile
            };

            full_entries.push((child_ino, kind, name));
        }

        for (i, (ino, kind, name)) in full_entries.iter().enumerate().skip(offset as usize) {
            if reply.add(*ino, (i + 1) as i64, *kind, name) {
                break;
            }
        }
        reply.ok();
    }

    fn open(&mut self, _req: &Request, ino: u64, flags: i32, reply: ReplyOpen) {
        let backing = match self.inode_to_backing_path(ino) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        if !backing.exists() {
            reply.error(libc::ENOENT);
            return;
        }

        let fh = self.alloc_fh();
        self.open_files.write().insert(
            fh,
            OpenFile {
                path: backing,
                flags,
            },
        );
        reply.opened(fh, 0);
    }

    fn read(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        self.metrics.fuse_reads_total.inc();

        let backing = match self.inode_to_backing_path(ino) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        match std::fs::read(&backing) {
            Ok(data) => {
                let start = offset as usize;
                if start >= data.len() {
                    reply.data(&[]);
                } else {
                    let end = (start + size as usize).min(data.len());
                    reply.data(&data[start..end]);
                }
            }
            Err(e) => {
                reply.error(e.raw_os_error().unwrap_or(libc::EIO));
            }
        }
    }

    fn write(
        &mut self,
        _req: &Request,
        ino: u64,
        _fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyWrite,
    ) {
        self.metrics.fuse_writes_total.inc();

        let backing = match self.inode_to_backing_path(ino) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        // Read existing content, apply write at offset, save
        let mut content = std::fs::read(&backing).unwrap_or_default();
        let offset = offset as usize;

        if offset > content.len() {
            content.resize(offset, 0);
        }

        let end = offset + data.len();
        if end > content.len() {
            content.resize(end, 0);
        }
        content[offset..end].copy_from_slice(data);

        match std::fs::write(&backing, &content) {
            Ok(()) => {
                reply.written(data.len() as u32);

                // Queue async indexing — NEVER block the write
                if let Some(mount_path) = self.inode_to_mount_path(ino) {
                    self.queue_index(mount_path);
                }
            }
            Err(e) => {
                reply.error(e.raw_os_error().unwrap_or(libc::EIO));
            }
        }
    }

    fn create(
        &mut self,
        _req: &Request,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        flags: i32,
        reply: ReplyCreate,
    ) {
        let parent_backing = match self.inode_to_backing_path(parent) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let child_backing = parent_backing.join(name);

        // Create the file
        match std::fs::File::create(&child_backing) {
            Ok(_) => {
                let parent_rel = self
                    .inode_map
                    .read()
                    .get(&parent)
                    .cloned()
                    .unwrap_or_default();
                let child_rel = parent_rel.join(name);
                let ino = self.register_inode(child_rel);

                match std::fs::metadata(&child_backing) {
                    Ok(meta) => {
                        let attr = self.meta_to_attr(ino, &meta);
                        let fh = self.alloc_fh();
                        self.open_files.write().insert(
                            fh,
                            OpenFile {
                                path: child_backing,
                                flags,
                            },
                        );
                        reply.created(&TTL, &attr, 0, fh, 0);
                    }
                    Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
                }
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn mkdir(
        &mut self,
        _req: &Request,
        parent: u64,
        name: &OsStr,
        mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let parent_backing = match self.inode_to_backing_path(parent) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let child_backing = parent_backing.join(name);

        match std::fs::create_dir(&child_backing) {
            Ok(()) => {
                let parent_rel = self
                    .inode_map
                    .read()
                    .get(&parent)
                    .cloned()
                    .unwrap_or_default();
                let child_rel = parent_rel.join(name);
                let ino = self.register_inode(child_rel);

                match std::fs::metadata(&child_backing) {
                    Ok(meta) => {
                        let attr = self.meta_to_attr(ino, &meta);
                        reply.entry(&TTL, &attr, 0);
                    }
                    Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
                }
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn unlink(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let parent_backing = match self.inode_to_backing_path(parent) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let child_backing = parent_backing.join(name);

        match std::fs::remove_file(&child_backing) {
            Ok(()) => {
                // Queue removal from index
                let parent_rel = self
                    .inode_map
                    .read()
                    .get(&parent)
                    .cloned()
                    .unwrap_or_default();
                let child_rel = parent_rel.join(name);
                let mount_path = self.mount_point.join(&child_rel);
                let _ = self.index_tx.try_send(IndexJob::Remove(mount_path));
                reply.ok();
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn rmdir(&mut self, _req: &Request, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let parent_backing = match self.inode_to_backing_path(parent) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let child_backing = parent_backing.join(name);

        match std::fs::remove_dir(&child_backing) {
            Ok(()) => reply.ok(),
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn rename(
        &mut self,
        _req: &Request,
        parent: u64,
        name: &OsStr,
        newparent: u64,
        newname: &OsStr,
        _flags: u32,
        reply: ReplyEmpty,
    ) {
        let src_parent = match self.inode_to_backing_path(parent) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };
        let dst_parent = match self.inode_to_backing_path(newparent) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        let src = src_parent.join(name);
        let dst = dst_parent.join(newname);

        match std::fs::rename(&src, &dst) {
            Ok(()) => {
                // Queue removal of old path and indexing of new path
                let src_rel = self
                    .inode_map
                    .read()
                    .get(&parent)
                    .cloned()
                    .unwrap_or_default();
                let old_mount = self.mount_point.join(src_rel.join(name));
                let dst_rel = self
                    .inode_map
                    .read()
                    .get(&newparent)
                    .cloned()
                    .unwrap_or_default();
                let new_mount = self.mount_point.join(dst_rel.join(newname));

                let _ = self.index_tx.try_send(IndexJob::Remove(old_mount));
                let _ = self.index_tx.try_send(IndexJob::Update(new_mount));

                reply.ok();
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }

    fn release(
        &mut self,
        _req: &Request,
        _ino: u64,
        fh: u64,
        _flags: i32,
        _lock_owner: Option<u64>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        self.open_files.write().remove(&fh);
        reply.ok();
    }

    fn setattr(
        &mut self,
        _req: &Request,
        ino: u64,
        _mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        size: Option<u64>,
        _atime: Option<fuser::TimeOrNow>,
        _mtime: Option<fuser::TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        let backing = match self.inode_to_backing_path(ino) {
            Some(p) => p,
            None => {
                reply.error(libc::ENOENT);
                return;
            }
        };

        // Handle truncate
        if let Some(new_size) = size {
            let file = match std::fs::OpenOptions::new().write(true).open(&backing) {
                Ok(f) => f,
                Err(e) => {
                    reply.error(e.raw_os_error().unwrap_or(libc::EIO));
                    return;
                }
            };
            if let Err(e) = file.set_len(new_size) {
                reply.error(e.raw_os_error().unwrap_or(libc::EIO));
                return;
            }
        }

        match std::fs::metadata(&backing) {
            Ok(meta) => {
                let attr = self.meta_to_attr(ino, &meta);
                reply.attr(&TTL, &attr);
            }
            Err(e) => reply.error(e.raw_os_error().unwrap_or(libc::EIO)),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_fuse_module_compiles() {
        // This test verifies the FUSE module compiles with the feature flag
        assert!(true);
    }
}
