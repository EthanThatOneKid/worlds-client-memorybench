import { createHash } from "node:crypto"
import { mkdir, readFile, writeFile } from "node:fs/promises"
import { dirname, join } from "node:path"
import type { EmbeddingService } from "@worlds/client/search-index/embedding-service"
import { logger } from "../../utils/logger"

const CACHE_ROOT = join(process.cwd(), "data", "cache", "embeddings")

interface CachedVector {
  dims: number
  values: number[]
}

function fileFor(label: string, hash: string): string {
  return join(CACHE_ROOT, label, `${hash}.json`)
}

async function tryRead(file: string): Promise<Float32Array | null> {
  try {
    const raw = await readFile(file, "utf8")
    const parsed = JSON.parse(raw) as CachedVector
    if (!Array.isArray(parsed.values) || typeof parsed.dims !== "number") return null
    return Float32Array.from(parsed.values)
  } catch {
    return null
  }
}

async function writeCache(file: string, vec: Float32Array | number[]): Promise<void> {
  const arr = vec instanceof Float32Array ? vec : Float32Array.from(vec)
  await mkdir(dirname(file), { recursive: true })
  await writeFile(file, JSON.stringify({ dims: arr.length, values: Array.from(arr) }), "utf8")
}

/**
 * CachedEmbeddingService wraps any EmbeddingService with a content-addressed
 * on-disk cache under `data/cache/embeddings/{label}/`. Entries are keyed by
 * `sha256(text)` under a per-service label (`{provider}/{model}`) and are
 * immutable — never overwritten. A corrupt or mismatched entry is treated as
 * a miss and re-embedded (self-invalidating). The cache is shared across
 * runs (not per-containerTag), so a fresh run ID re-embeds nothing.
 * `--force` / `clear()` never touch this cache; use the `cache-clear`
 * command for an explicit reset.
 */
export class CachedEmbeddingService implements EmbeddingService {
  private readonly inner: EmbeddingService
  private readonly label: string

  constructor(inner: EmbeddingService, label: string) {
    this.inner = inner
    this.label = label
  }

  async embed(texts: string[]): Promise<Array<Float32Array | number[]>> {
    if (texts.length === 0) return []

    const hashes = texts.map((t) => createHash("sha256").update(t).digest("hex"))
    const vectors: (Float32Array | null)[] = new Array(texts.length).fill(null)
    const missing: number[] = []

    await Promise.all(
      hashes.map(async (hash, i) => {
        const cached = await tryRead(fileFor(this.label, hash))
        if (cached) vectors[i] = cached
        else missing.push(i)
      })
    )

    if (missing.length > 0) {
      const fresh = await this.inner.embed(missing.map((i) => texts[i]))
      await Promise.all(
        missing.map(async (index, k) => {
          const vec = fresh[k]
          const arr = vec instanceof Float32Array ? vec : Float32Array.from(vec)
          vectors[index] = arr
          await writeCache(fileFor(this.label, hashes[index]), arr)
        })
      )
      logger.debug(
        `CachedEmbeddingService[${this.label}]: ${texts.length - missing.length}/${texts.length} hits, ${missing.length} embedded fresh`
      )
    } else {
      logger.debug(
        `CachedEmbeddingService[${this.label}]: ${texts.length}/${texts.length} cache hits`
      )
    }

    return vectors as Float32Array[]
  }
}
