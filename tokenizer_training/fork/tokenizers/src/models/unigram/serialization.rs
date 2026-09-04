use super::model::Unigram;
use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

impl Serialize for Unigram {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // NOTE: we now write 5 fields: type, unk_id, vocab, byte_fallback, unit_cost
        let mut model = serializer.serialize_struct("Unigram", 5)?;

        model.serialize_field("type", "Unigram")?;
        model.serialize_field("unk_id", &self.unk_id)?;                  // stored as-is
        model.serialize_field("vocab", &self.vocab)?;                    // (token, score)
        model.serialize_field("byte_fallback", &self.byte_fallback())?;  // use getter
        model.serialize_field("unit_cost", &self.unit_cost())?;          // NEW: persist unit-cost mode

        model.end()
    }
}

impl<'de> Deserialize<'de> for Unigram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Add "unit_cost" to the expected fields; default to false if absent (back-compat).
        deserializer.deserialize_struct(
            "Unigram",
            &["type", "vocab", "unk_id", "byte_fallback", "unit_cost"],
            UnigramVisitor,
        )
    }
}

struct UnigramVisitor;

impl<'de> Visitor<'de> for UnigramVisitor {
    type Value = Unigram;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct Unigram")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut vocab: Option<Vec<(String, f64)>> = None;
        let mut unk_id: Option<usize> = None;
        let mut byte_fallback: bool = false;
        let mut unit_cost: bool = false; // NEW: default false for old files

        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "unk_id" => {
                    unk_id = map.next_value()?;
                }
                "byte_fallback" => {
                    byte_fallback = map.next_value()?;
                }
                "unit_cost" => {
                    unit_cost = map.next_value()?; // NEW: read the flag if present
                }
                "vocab" => {
                    vocab = Some(map.next_value()?);
                }
                "type" => match map.next_value()? {
                    "Unigram" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"Unigram",
                        ))
                    }
                },
                _ => (), // ignore unknown keys for forward-compat
            }
        }

        match (vocab, unk_id, byte_fallback) {
            (Some(vocab), unk_id, byte_fallback) => {
                // First build the model with standard constructor (unit_cost defaults to false)…
                let mut model = Unigram::from(vocab, unk_id, byte_fallback)
                    .map_err(|err| Error::custom(format!("Unable to load vocab {err:?}")))?;
                // …then apply the unit-cost flag (public setter keeps encapsulation).
                model.set_unit_cost(unit_cost);
                Ok(model)
            }
            (None, _, _) => Err(Error::custom("Missing vocab")),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_serialization() {
        let vocab = vec![("<unk>".to_string(), 0.0), ("a".to_string(), -0.5)];
        let model = Unigram::from(vocab, Some(0), false).unwrap();

        let data = serde_json::to_string(&model).unwrap();
        let reconstructed: Unigram = serde_json::from_str(&data).unwrap();

        // Equality ignores unit_cost flag by design (same as upstream: compares unk_id & vocab)
        assert_eq!(model, reconstructed);
    }

    #[test]
    fn test_serialization_unk_id_not_zero() {
        let vocab = vec![("a".to_string(), -0.5), ("<unk>".to_string(), 0.0)];
        let model = Unigram::from(vocab, Some(1), false).unwrap();

        let data = serde_json::to_string(&model).unwrap();
        let reconstructed: Unigram = serde_json::from_str(&data).unwrap();

        assert_eq!(model, reconstructed);
    }

    #[test]
    fn test_serialization_no_unk_id() {
        let vocab = vec![("a".to_string(), -0.5)];
        let model = Unigram::from(vocab, None, false).unwrap();

        let data = serde_json::to_string(&model).unwrap();
        let reconstructed: Unigram = serde_json::from_str(&data).unwrap();

        assert_eq!(model, reconstructed);
    }

    #[test]
    fn test_serialization_with_unit_cost_flag() {
        // NEW: round-trip when unit_cost is toggled
        let vocab = vec![("<unk>".to_string(), 0.0), ("ab".to_string(), -1.2)];
        let mut model = Unigram::from(vocab, Some(0), false).unwrap();
        model.set_unit_cost(true);

        let data = serde_json::to_string(&model).unwrap();
        let reconstructed: Unigram = serde_json::from_str(&data).unwrap();

        // Behavior check: flag preserved after round-trip
        assert!(reconstructed.unit_cost());
        // Structural equality per PartialEq still holds (compares unk_id & vocab only)
        assert_eq!(model, reconstructed);
    }
}