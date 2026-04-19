import type { ClassificationHit, SearchResult, VocabMode } from "./types";

const MOCK_CATALOG: SearchResult[] = [
  {
    id: "m1",
    recordingId: "894721",
    commonName: "Black-capped Chickadee",
    scientificName: "Poecile atricapillus",
    vocalizationType: "song",
    duration: "0:42",
    speciesCode: "black_capped_chickadee",
    similarity: 0.94,
    imageUrl: placeholderAvatar("Black-capped Chickadee"),
  },
  {
    id: "m2",
    recordingId: "120334",
    commonName: "American Robin",
    scientificName: "Turdus migratorius",
    vocalizationType: "call",
    duration: "1:05",
    speciesCode: "american_robin",
    similarity: 0.89,
    imageUrl: placeholderAvatar("American Robin"),
  },
  {
    id: "m3",
    recordingId: "551902",
    commonName: "Song Sparrow",
    scientificName: "Melospiza melodia",
    vocalizationType: "song",
    duration: "0:38",
    speciesCode: "song_sparrow",
    similarity: 0.85,
    imageUrl: placeholderAvatar("Song Sparrow"),
  },
  {
    id: "m4",
    recordingId: "210887",
    commonName: "Northern Cardinal",
    scientificName: "Cardinalis cardinalis",
    vocalizationType: "call",
    duration: "0:22",
    speciesCode: "northern_cardinal",
    similarity: 0.81,
    imageUrl: placeholderAvatar("Northern Cardinal"),
  },
  {
    id: "m5",
    recordingId: "334411",
    commonName: "Blue Jay",
    scientificName: "Cyanocitta cristata",
    vocalizationType: "call, song",
    duration: "0:55",
    speciesCode: "blue_jay",
    similarity: 0.78,
    imageUrl: placeholderAvatar("Blue Jay"),
  },
  {
    id: "m6",
    recordingId: "778201",
    commonName: "Red-winged Blackbird",
    scientificName: "Agelaius phoeniceus",
    vocalizationType: "song",
    duration: "0:31",
    speciesCode: "red_winged_blackbird",
    similarity: 0.76,
    imageUrl: placeholderAvatar("Red-winged Blackbird"),
  },
  {
    id: "m7",
    recordingId: "445566",
    commonName: "Common Yellowthroat",
    scientificName: "Geothlypis trichas",
    vocalizationType: "song",
    duration: "0:48",
    speciesCode: "common_yellowthroat",
    similarity: 0.72,
    imageUrl: placeholderAvatar("Common Yellowthroat"),
  },
  {
    id: "m8",
    recordingId: "990011",
    commonName: "White-throated Sparrow",
    scientificName: "Zonotrichia albicollis",
    vocalizationType: "song",
    duration: "1:12",
    speciesCode: "white_throated_sparrow",
    similarity: 0.69,
    imageUrl: placeholderAvatar("White-throated Sparrow"),
  },
  {
    id: "m9",
    recordingId: "223344",
    commonName: "Downy Woodpecker",
    scientificName: "Dryobates pubescens",
    vocalizationType: "drumming",
    duration: "0:18",
    speciesCode: "downy_woodpecker",
    similarity: 0.66,
    imageUrl: placeholderAvatar("Downy Woodpecker"),
  },
  {
    id: "m10",
    recordingId: "667788",
    commonName: "Canada Goose",
    scientificName: "Branta canadensis",
    vocalizationType: "honk",
    duration: "0:45",
    speciesCode: "canada_goose",
    similarity: 0.63,
    imageUrl: placeholderAvatar("Canada Goose"),
  },
  {
    id: "m11",
    recordingId: "112233",
    commonName: "Mallard",
    scientificName: "Anas platyrhynchos",
    vocalizationType: "quack",
    duration: "0:28",
    speciesCode: "mallard",
    similarity: 0.61,
    imageUrl: placeholderAvatar("Mallard"),
  },
  {
    id: "m12",
    recordingId: "445577",
    commonName: "Belted Kingfisher",
    scientificName: "Megaceryle alcyon",
    vocalizationType: "rattle",
    duration: "0:35",
    speciesCode: "belted_kingfisher",
    similarity: 0.58,
    imageUrl: placeholderAvatar("Belted Kingfisher"),
  },
  {
    id: "m13",
    recordingId: "889900",
    commonName: "Barred Owl",
    scientificName: "Strix varia",
    vocalizationType: "hoot",
    duration: "0:52",
    speciesCode: "barred_owl",
    similarity: 0.55,
    imageUrl: placeholderAvatar("Barred Owl"),
  },
  {
    id: "m14",
    recordingId: "334455",
    commonName: "Ruby-throated Hummingbird",
    scientificName: "Archilochus colubris",
    vocalizationType: "chip",
    duration: "0:15",
    speciesCode: "ruby_throated_hummingbird",
    similarity: 0.52,
    imageUrl: placeholderAvatar("Ruby-throated Hummingbird"),
  },
  {
    id: "m15",
    recordingId: "778899",
    commonName: "Great Blue Heron",
    scientificName: "Ardea herodias",
    vocalizationType: "squawk",
    duration: "0:41",
    speciesCode: "great_blue_heron",
    similarity: 0.49,
    imageUrl: placeholderAvatar("Great Blue Heron"),
  },
];

function placeholderAvatar(name: string): string {
  const q = encodeURIComponent(name);
  return `https://ui-avatars.com/api/?name=${q}&size=128&background=1a5f4a&color=fff&bold=true`;
}

function delay<T>(ms: number, value: T): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

function normalize(s: string): string {
  return s.trim().toLowerCase();
}

/**
 * Mock search against the indexed dataset. Filters by substring on common or scientific name.
 */
export async function searchDataset(
  query: string,
  vocab: VocabMode
): Promise<SearchResult[]> {
  const q = normalize(query);
  const list = MOCK_CATALOG.filter((row) => {
    if (!q) return true;
    const primary =
      vocab === "scientific" ? row.scientificName : row.commonName;
    const secondary =
      vocab === "scientific" ? row.commonName : row.scientificName;
    return (
      normalize(primary).includes(q) ||
      normalize(secondary).includes(q) ||
      normalize(row.speciesCode).includes(q)
    );
  });
  return delay(400, list.length ? list : MOCK_CATALOG.slice(0, 8));
}

export async function classifyUpload(_file: File): Promise<ClassificationHit[]> {
  const hits: ClassificationHit[] = [
    {
      label: "Black-capped Chickadee",
      scientificName: "Poecile atricapillus",
      score: 0.41,
    },
    { label: "Carolina Chickadee", scientificName: "Poecile carolinensis", score: 0.22 },
    { label: "Boreal Chickadee", scientificName: "Poecile hudsonicus", score: 0.09 },
    { label: "Chestnut-backed Chickadee", scientificName: "Poecile rufescens", score: 0.06 },
    { label: "Other / background", score: 0.05 },
  ];
  return delay(600, hits);
}

export async function searchSimilarToUpload(_file: File): Promise<SearchResult[]> {
  return delay(500, MOCK_CATALOG.slice(0, 6).map((r, i) => ({
    ...r,
    similarity: 0.92 - i * 0.07,
  })));
}

export function getResultById(id: string): SearchResult | undefined {
  return MOCK_CATALOG.find((r) => r.id === id);
}
