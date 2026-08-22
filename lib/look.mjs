/**
 * A Look is the design: what every region of a Blank is wearing.
 *
 * It is plain data and every function here returns a new value rather than mutating
 * one, which is what lets the studio keep an undo stack for free and lets the baker
 * be a pure function of (Blank, Look). The Blank is immutable once built; the Look is
 * the only thing a person actually edits.
 *
 * A Look never contains geometry, file paths or texture bytes. It contains region ids
 * and material ids, so it survives a rebuild of the Blank and is small enough to keep
 * in a URL if it ever needs to be.
 */
import { getMaterial } from './materials.mjs';

export const DEFAULT_COAT = Object.freeze({
  materialId: 'satin',
  tint: '#b9bed6',
  tileScale: 1,
  roughness: null,   // null means "whatever the material says"
  metallic: null,
  glow: true,        // photo material only: ship the photographed glow as emission
});

/** Everything wearing the default. */
export function emptyLook(regionCount) {
  return {
    version: 1,
    coats: Array.from({ length: regionCount }, () => ({ ...DEFAULT_COAT })),
  };
}

/** Change one region. Returns a new Look; the original is untouched. */
export function setCoat(look, regionId, patch) {
  if (regionId < 0 || regionId >= look.coats.length) return look;
  const coats = look.coats.slice();
  coats[regionId] = { ...coats[regionId], ...patch };
  return { ...look, coats };
}

/** Change every region at once, which is what "apply to all" wants. */
export function setAll(look, patch) {
  return { ...look, coats: look.coats.map((c) => ({ ...c, ...patch })) };
}

/**
 * Group regions that are wearing exactly the same thing.
 *
 * This is what makes the export cheap: twenty-four regions in four colourways bakes
 * to four materials and four triangle groups, not twenty-four of each. Both glTF and
 * USD pay per material, and AR Quick Look in particular gets noticeably slower with
 * a long material list.
 */
export function groups(look) {
  const byKey = new Map();
  look.coats.forEach((coat, regionId) => {
    const key = coatKey(coat);
    const g = byKey.get(key);
    if (g) g.regions.push(regionId);
    else byKey.set(key, { key, coat, regions: [regionId] });
  });
  return [...byKey.values()];
}

export function coatKey(c) {
  return [c.materialId, c.tint.toLowerCase(), c.tileScale, c.roughness ?? 'm', c.metallic ?? 'm', c.glow === false ? 'noglow' : 'glow'].join('|');
}

/**
 * Reject a Look that cannot be baked, with a reason a person can act on.
 * Called on the way in from the browser, where the data is untrusted.
 */
export function validate(look, regionCount) {
  if (!look || typeof look !== 'object') return 'not an object';
  if (!Array.isArray(look.coats)) return 'no coats array';
  if (look.coats.length !== regionCount) {
    return `look covers ${look.coats.length} regions, blank has ${regionCount}`;
  }
  for (const [i, c] of look.coats.entries()) {
    if (!getMaterial(c.materialId)) return `region ${i}: unknown material "${c.materialId}"`;
    if (!/^#[0-9a-fA-F]{6}$/.test(c.tint || '')) return `region ${i}: tint must be #rrggbb`;
    if (!(c.tileScale > 0.05 && c.tileScale <= 20)) return `region ${i}: tileScale out of range`;
    for (const k of ['roughness', 'metallic']) {
      if (c[k] !== null && c[k] !== undefined && !(c[k] >= 0 && c[k] <= 1)) {
        return `region ${i}: ${k} must be 0..1 or null`;
      }
    }
  }
  return null;
}

/** #rrggbb to linear-space float triple, which is what both glTF and USD want. */
export function tintToLinear(hex) {
  const n = parseInt(hex.slice(1), 16);
  const srgb = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => v / 255);
  // the usual sRGB electro-optical transfer function, not the 2.2 approximation:
  // the difference is visible in the dark end, which is where paint colours live
  return srgb.map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
}
