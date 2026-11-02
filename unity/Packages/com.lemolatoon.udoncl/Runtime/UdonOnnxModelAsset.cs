using UnityEngine;

namespace Lemolatoon.Udoncl
{
    /// <summary>
    /// Stable, model-independent Unity representation of a compiled Udon ONNX model.
    /// </summary>
    public sealed class UdonOnnxModelAsset : ScriptableObject
    {
        [SerializeField]
        private TextAsset model;

        [SerializeField]
        private TextAsset executionPlan;

        [SerializeField]
        private string executionProvider;

        [SerializeField]
        private string profile;

        [SerializeField]
        private Object hostProgram;

        [SerializeField]
        private Material[] passMaterials;

        [SerializeField]
        private int textureArenaTotalWords;

        [SerializeField]
        private int[] textureSlotCapacities;

        [SerializeField]
        private int[] textureSlotWidths;

        [SerializeField]
        private int[] textureSlotHeights;

        [SerializeField]
        private RenderTexture[] textureArenaSlots;

        [SerializeField]
        private Texture2D[] staticInputTextures;

        [SerializeField]
        private string[] inputNames;

        [SerializeField]
        private string[] inputDataTypes;

        [SerializeField]
        private int[] inputWords;

        [SerializeField]
        private int[] inputSlots;

        [SerializeField]
        private string[] outputNames;

        [SerializeField]
        private string[] outputDataTypes;

        [SerializeField]
        private int[] outputWords;

        [SerializeField]
        private int[] outputSlots;

        [SerializeField]
        private GameObject runtimePrefab;

        public TextAsset Model => model;

        public TextAsset ExecutionPlan => executionPlan;

        public string ExecutionProvider => executionProvider;

        public string Profile => profile;

        public Object HostProgram => hostProgram;

        public Material[] PassMaterials => passMaterials;

        public int TextureArenaTotalWords => textureArenaTotalWords;

        public int[] TextureSlotCapacities => textureSlotCapacities;

        public int[] TextureSlotWidths => textureSlotWidths;

        public int[] TextureSlotHeights => textureSlotHeights;

        /// <summary>
        /// Reusable RGBA32 word-storage surfaces indexed by the execution plan's arena slot.
        /// </summary>
        public RenderTexture[] TextureArenaSlots => textureArenaSlots;

        public Texture2D[] StaticInputTextures => staticInputTextures;

        public string[] InputNames => inputNames;

        public string[] InputDataTypes => inputDataTypes;

        public int[] InputWords => inputWords;

        public int[] InputSlots => inputSlots;

        public string[] OutputNames => outputNames;

        public string[] OutputDataTypes => outputDataTypes;

        public int[] OutputWords => outputWords;

        public int[] OutputSlots => outputSlots;

        public GameObject RuntimePrefab => runtimePrefab;
    }
}
