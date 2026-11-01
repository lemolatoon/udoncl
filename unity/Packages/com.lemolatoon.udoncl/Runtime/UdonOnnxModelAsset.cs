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
    }
}
