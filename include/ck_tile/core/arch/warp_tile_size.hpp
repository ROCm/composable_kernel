// ============================================================================
// Architecture-specific parameter definitions
// We'll define all parameters for all supported architectures here.
// ============================================================================

// Parameters for gfx120x (using a namespace for organization or just global constexpr)
namespace Gfx120x {
constexpr ck_tile::index_t WarpTile = 32;
}

// Parameters for gfx90x (example values, adjust as needed)
namespace Gfx90x {
constexpr ck_tile::index_t WarpTile = 64;
}

// Generic Parameters - should never be used in this example
// templated run function should only be instantiated for Gfx120x and Gfx90x
namespace Generic {
constexpr ck_tile::index_t WarpTile = -1;
}
