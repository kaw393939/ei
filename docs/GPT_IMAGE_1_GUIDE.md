# GPT-Image-1 Guide

OpenAI's `gpt-image-1` is their latest image generation model, designed for high-quality, flexible image creation with advanced features not available in previous models like DALL-E 3.

## Quick Reference

| Feature | Options |
|---------|---------|
| **Sizes** | `1024x1024`, `1024x1536`, `1536x1024`, `auto` |
| **Quality** | `low`, `medium`, `high`, `auto` |
| **Formats** | `png`, `jpeg`, `webp` |
| **Background** | `transparent`, `default` |
| **Delivery** | `url` (60-min expiry), `b64_json` (base64) |

## Image Sizes

gpt-image-1 supports three predefined sizes plus automatic selection:

- **1024x1024**: Square format (1:1 ratio)
- **1024x1536**: Portrait format (2:3 ratio)
- **1536x1024**: Landscape format (3:2 ratio)
- **auto**: Model chooses optimal size based on prompt


## Quality Levels & Pricing

gpt-image-1 offers three quality tiers with transparent pricing:

| Quality | Cost per Image | Best For |
|---------|---------------|----------|
| **low** | $0.01 | Quick drafts, iterations, testing |
| **medium** | $0.04 | Standard quality, most use cases |
| **high** | $0.17 | Final outputs, professional use |
| **auto** | Variable | Model determines optimal quality |

**Note**: Pricing is per square (1024x1024) image. Non-square images may have proportional costs.


## Output Formats

### PNG (Default)

- **Supports transparency**: Set `background: "transparent"` for images with transparent backgrounds
- **Best for**: Images with transparency needs, high quality requirements
- **Compression**: Lossless


### JPEG

- **No transparency**: Always has opaque background
- **Best for**: Photos, web images where transparency isn't needed
- **Compression**: Lossy, smaller file sizes


### WebP

- **Compression control**: Accepts `quality` parameter (0-100)
- **Supports transparency**: Can have transparent backgrounds
- **Best for**: Modern web applications, optimal size/quality balance


## Transparent Backgrounds

One of gpt-image-1's key features is native transparent background support:

```python
# Example API call for transparent background
response = client.images.generate(
    model="gpt-image-1",
    prompt="A red apple on a branch",
    format="png",
    background="transparent"
)
```

**Use cases**:

- Product images for e-commerce
- Icons and logos
- UI elements
- Overlay graphics
- Marketing materials


## Image Editing Features

gpt-image-1 supports advanced editing capabilities:

### Multiple Input Images

- **Up to 10 images**: Provide multiple reference images
- **Use case**: Combine elements, maintain consistency, style transfer


### Masking

- **Selective editing**: Use masks to specify which areas to modify
- **Precision control**: Edit specific regions while preserving others


### Input Fidelity

- **Adherence control**: Adjust how closely output matches input images
- **Range**: Typically 0.0-1.0
- **Use case**: Balance between prompt creativity and input preservation


## Key Improvements Over DALL-E 3

1. **Enhanced Text Rendering**: Superior ability to generate accurate text within images
2. **Flexible Formats**: PNG, JPEG, WebP support with quality controls

3. **Transparent Backgrounds**: Native support without workarounds
4. **Quality Tiers**: Cost-effective options for different use cases
5. **Advanced Editing**: Multi-image inputs and masking capabilities
6. **Metadata**: C2PA provenance metadata support for authenticity

## Model Differences: gpt-image-1 vs DALL-E 3

| Feature | gpt-image-1 | DALL-E 3 |
|---------|-------------|----------|
| Text in images | ✅ Improved | ⚠️ Limited |
| Transparent backgrounds | ✅ Native | ❌ Not available |
| Output formats | PNG, JPEG, WebP | PNG only |
| Quality tiers | 3 levels + auto | Standard/HD |
| Editing features | ✅ Masks, multi-input | ⚠️ Basic |
| Style control | Prompt-based | vivid/natural toggle |
| Pricing model | Per-quality tiers | Standard/HD pricing |

**Note**: DALL-E 3's `style` parameter (`vivid`/`natural`) is not available in gpt-image-1. Use detailed prompts instead.


## Best Practices

### 1. Quality Selection

- Start with **low** for rapid iteration and concept testing
- Use **medium** for most production use cases
- Reserve **high** for final outputs requiring maximum detail
- Use **auto** when unsure (model optimizes based on prompt)


### 2. Size Selection

- **Square (1024x1024)**: Social media, profiles, thumbnails
- **Portrait (1024x1536)**: Mobile screens, vertical displays
- **Landscape (1536x1024)**: Website headers, presentations
- **Auto**: When aspect ratio isn't critical


### 3. Format Selection

- **PNG**: When transparency is needed or quality is paramount
- **JPEG**: For photos and when file size matters (web)
- **WebP**: Modern applications needing size/quality balance


### 4. Prompt Engineering

- Be specific about desired elements, style, and composition
- Mention text explicitly if needed in the image
- Specify background requirements (transparent vs colored)
- Include lighting, perspective, and mood details


### 5. Cost Optimization

- Test prompts at **low** quality first
- Use **medium** for production unless detail demands **high**
- Consider WebP format for smaller file sizes
- Use appropriate size—don't request larger than needed


## Common Use Cases

### Product Photography

```yaml
quality: high
size: 1024x1024
format: png
background: transparent
```

### Social Media Graphics

```yaml
quality: medium
size: 1024x1024  # Instagram, Facebook
format: jpeg
background: default
```

### Website Headers

```yaml
quality: medium
size: 1536x1024
format: webp
background: default
```

### UI Icons

```yaml
quality: medium
size: 1024x1024
format: png
background: transparent
```

### Marketing Materials

```yaml
quality: high
size: 1536x1024 or 1024x1536
format: png
background: transparent or default
```

## API Reference

### Basic Generation

```python
from openai import OpenAI

client = OpenAI()

response = client.images.generate(
    model="gpt-image-1",
    prompt="Your detailed prompt here",
    size="1024x1024",
    quality="medium",
    format="png",
    background="default",
    response_format="url"  # or "b64_json"
)

image_url = response.data[0].url
```

### With Transparent Background

```python
response = client.images.generate(
    model="gpt-image-1",
    prompt="A logo for a tech company",
    format="png",
    background="transparent",
    quality="high"
)
```

### WebP with Compression

```python
response = client.images.generate(
    model="gpt-image-1",
    prompt="A landscape scene",
    format="webp",
    quality=85,  # 0-100, compression level
    size="1536x1024"
)
```

## Using with EI CLI

The EI CLI uses gpt-image-1 for all image generation commands:

```bash
# Basic image generation
eai image "A serene mountain landscape at sunset"

# Specify quality (defaults to medium)
eai image "Product photo of headphones" --quality high

# Request transparent background
eai image "Company logo" --transparent

# Custom size
eai image "Website banner" --size 1536x1024
```

## Troubleshooting

### Issue: Generated text is unclear

**Solution**: Be very explicit in prompt: "Large bold text saying 'HELLO' in Arial font"


### Issue: Colors not matching request

**Solution**: Use specific color names or hex codes in prompt: "vibrant red (#FF0000)"


### Issue: Transparent background has artifacts

**Solution**: Ensure format is PNG, not JPEG. WebP also supports transparency.


### Issue: Image quality lower than expected

**Solution**: Upgrade from `low` to `medium` or `high` quality tier


### Issue: File size too large

**Solution**: Use JPEG or WebP format with compression instead of PNG


## Resources

- **OpenAI API Documentation**: [platform.openai.com/docs/api-reference/images](https://platform.openai.com/docs/api-reference/images)
- **Cost Calculator**: Track costs by monitoring quality levels used
- **Community Examples**: OpenAI community forum for prompt inspiration


## Version History

- **gpt-image-1**: Current latest model (2024)
- Previous: DALL-E 3, DALL-E 2


---


**Last Updated**: January 2025
**Model**: gpt-image-1
**Status**: Production-ready
